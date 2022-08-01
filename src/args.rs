use core::fmt;
use std::cmp;
use std::env;
use std::error::Error;
use std::ffi::{OsStr, OsString};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process;
use std::sync::Arc;

use clap;
use grep::cli;
use grep::matcher::LineTerminator;
#[cfg(feature = "pcre2")]
use grep::pcre2::{
    RegexMatcher as PCRE2RegexMatcher, RegexMatcherBuilder as PCRE2RegexMatcherBuilder,
};
use grep::regex::{
    RegexMatcher as RustRegexMatcher, RegexMatcherBuilder as RustRegexMatcherBuilder,
};
use grep::searcher::{BinaryDetection, Encoding, MmapChoice, Searcher, SearcherBuilder};
use ignore::overrides::{Override, OverrideBuilder};
use ignore::types::{FileTypeDef, Types, TypesBuilder};
use ignore::{WalkBuilder, WalkParallel};
use log;
use num_cpus;
use regex;
use termcolor::{BufferWriter, ColorChoice};

use crate::app;
use crate::config;
use crate::ignore_message;
use crate::logger::Logger;
use crate::messages::{set_ignore_messages, set_messages};
use crate::patch::{Patch, PatchBuilder};
use crate::search::{PatternMatcher, SearchWorker, SearchWorkerBuilder};
use crate::subject::SubjectBuilder;
use crate::Result;

/// The command that ripgrep should execute based on the command line
/// configuration.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Command {
    /// Search using exactly one thread.
    Search,
    /// Search using possibly many threads.
    SearchParallel,
    /// The command line parameters suggest that a search should occur, but
    /// ripgrep knows that a match can never be found (e.g., no given patterns
    /// or --max-count=0).
    SearchNever,
    /// Show the files that would be searched, but don't actually search them,
    /// and use exactly one thread.
    Files,
    /// Show the files that would be searched, but don't actually search them,
    /// and perform directory traversal using possibly many threads.
    FilesParallel,
    /// List all file type definitions configured, including the default file
    /// types and any additional file types added to the command line.
    Types,
    /// Print the version of PCRE2 in use.
    PCRE2Version,
}

#[derive(Debug, Copy, Clone)]
struct ErrReplacementTextNotSet {}

impl fmt::Display for ErrReplacementTextNotSet {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "replacement text not set")
    }
}

impl Error for ErrReplacementTextNotSet {}

static ERR_NO_REPLACEMENT: ErrReplacementTextNotSet = ErrReplacementTextNotSet {};

impl Command {
    /// Returns true if and only if this command requires executing a search.
    fn is_search(&self) -> bool {
        use self::Command::*;

        match *self {
            Search | SearchParallel => true,
            SearchNever | Files | FilesParallel | Types | PCRE2Version => false,
        }
    }
}

/// The primary configuration object used throughout ripgrep. It provides a
/// high-level convenient interface to the provided command line arguments.
///
/// An `Args` object is cheap to clone and can be used from multiple threads
/// simultaneously.
#[derive(Clone, Debug)]
pub struct Args(Arc<ArgsImp>);

#[derive(Clone, Debug)]
struct ArgsImp {
    /// Mid-to-low level routines for extracting CLI arguments.
    matches: ArgMatches,
    /// The patterns provided at the command line and/or via the -f/--file
    /// flag. This may be empty.
    patterns: Vec<String>,
    /// A matcher built from the patterns.
    ///
    /// It's important that this is only built once, since building this goes
    /// through regex compilation and various types of analyses. That is, if
    /// you need many of these (one per thread, for example), it is better to
    /// build it once and then clone it.
    matcher: PatternMatcher,
    /// The paths provided at the command line. This is guaranteed to be
    /// non-empty. (If no paths are provided, then a default path is created.)
    paths: Vec<PathBuf>,
    /// Returns true if and only if `paths` had to be populated with a single
    /// default path.
    using_default_path: bool,
}

impl Args {
    /// Parse the command line arguments for this process.
    ///
    /// If a CLI usage error occurred, then exit the process and print a usage
    /// or error message. Similarly, if the user requested the version of
    /// ripgrep, then print the version and exit.
    ///
    /// Also, initialize a global logger.
    pub fn parse() -> Result<Args> {
        // We parse the args given on CLI. This does not include args from
        // the config. We use the CLI args as an initial configuration while
        // trying to parse config files. If a config file exists and has
        // arguments, then we re-parse argv, otherwise we just use the matches
        // we have here.
        let early_matches = ArgMatches::new(clap_matches(env::args_os())?);
        set_messages(!early_matches.is_present("no-messages"));
        set_ignore_messages(!early_matches.is_present("no-ignore-messages"));

        if let Err(err) = Logger::init() {
            return Err(format!("failed to initialize logger: {}", err).into());
        }
        if early_matches.is_present("trace") {
            log::set_max_level(log::LevelFilter::Trace);
        } else if early_matches.is_present("debug") {
            log::set_max_level(log::LevelFilter::Debug);
        } else {
            log::set_max_level(log::LevelFilter::Warn);
        }

        let matches = early_matches.reconfigure()?;
        // The logging level may have changed if we brought in additional
        // arguments from a configuration file, so recheck it and set the log
        // level as appropriate.
        if matches.is_present("trace") {
            log::set_max_level(log::LevelFilter::Trace);
        } else if matches.is_present("debug") {
            log::set_max_level(log::LevelFilter::Debug);
        } else {
            log::set_max_level(log::LevelFilter::Warn);
        }
        set_messages(!matches.is_present("no-messages"));
        set_ignore_messages(!matches.is_present("no-ignore-messages"));
        matches.to_args()
    }

    /// Return direct access to command line arguments.
    fn matches(&self) -> &ArgMatches {
        &self.0.matches
    }

    /// Return the patterns found in the command line arguments. This includes
    /// patterns read via the -f/--file flags.
    fn patterns(&self) -> &[String] {
        &self.0.patterns
    }

    /// Return the matcher builder from the patterns.
    fn matcher(&self) -> &PatternMatcher {
        &self.0.matcher
    }

    /// Return the paths found in the command line arguments. This is
    /// guaranteed to be non-empty. In the case where no explicit arguments are
    /// provided, a single default path is provided automatically.
    fn paths(&self) -> &[PathBuf] {
        &self.0.paths
    }

    /// Returns true if and only if `paths` had to be populated with a default
    /// path, which occurs only when no paths were given as command line
    /// arguments.
    pub fn using_default_path(&self) -> bool {
        self.0.using_default_path
    }
}

/// High level public routines for building data structures used by RipPatch and
/// ripgrep from command line arguments.
impl Args {
    /// Create a new buffer writer for multi-threaded printing.
    pub fn buffer_writer(&self) -> BufferWriter {
        BufferWriter::stdout(ColorChoice::Never)
    }

    /// Build a worker for executing searches.
    ///
    /// Search results are written to the given writer.
    pub fn search_worker<W: io::Write>(&self, wtr: W) -> Result<SearchWorker<W>> {
        let matches = self.matches();
        let matcher = self.matcher().clone();
        let printer = self.matches().printer_patch(wtr)?;
        let searcher = matches.searcher(self.paths())?;
        let mut builder = SearchWorkerBuilder::new();
        builder
            .preprocessor(matches.preprocessor())?
            .preprocessor_globs(matches.preprocessor_globs()?)
            .search_zip(matches.is_present("search-zip"))
            .binary_detection_implicit(matches.binary_detection_implicit())
            .binary_detection_explicit(matches.binary_detection_explicit());
        Ok(builder.build(matcher, searcher, printer))
    }

    /// Return a builder for constructing subjects. A subject represents a
    /// single unit of something to search. Typically, this corresponds to a
    /// file or a stream such as stdin.
    pub fn subject_builder(&self) -> SubjectBuilder {
        let mut builder = SubjectBuilder::new();
        builder.strip_dot_prefix(self.using_default_path());
        builder
    }

    /// Execute the given function with a writer to stdout that enables color
    /// support based on the command line configuration.
    pub fn stdout(&self) -> cli::StandardStream {
        if self.matches().is_present("line-buffered") {
            cli::stdout_buffered_line(ColorChoice::Never)
        } else if self.matches().is_present("block-buffered") {
            cli::stdout_buffered_block(ColorChoice::Never)
        } else {
            cli::stdout(ColorChoice::Never)
        }
    }

    /// Return the type definitions compiled into ripgrep.
    ///
    /// If there was a problem reading and parsing the type definitions, then
    /// this returns an error.
    pub fn type_defs(&self) -> Result<Vec<FileTypeDef>> {
        Ok(self.matches().types()?.definitions().to_vec())
    }

    /// Return a walker that never uses additional threads.
    /* XXX decide whether to support this
    pub fn walker(&self) -> Result<Walk> {
        Ok(self.matches().walker_builder(self.paths())?.build())
    }
    */

    /// Return a parallel walker that may use additional threads.
    pub fn walker_parallel(&self) -> Result<WalkParallel> {
        Ok(self
            .matches()
            .walker_builder(self.paths())?
            .build_parallel())
    }
}

/// `ArgMatches` wraps `clap::ArgMatches` and provides semantic meaning to
/// the parsed arguments.
#[derive(Clone, Debug)]
struct ArgMatches(clap::ArgMatches<'static>);

/// Encoding mode the searcher will use.
#[derive(Clone, Debug)]
enum EncodingMode {
    /// Use an explicit encoding forcefully, but let BOM sniffing override it.
    Some(Encoding),
    /// Use only BOM sniffing to auto-detect an encoding.
    Auto,
    /// Use no explicit encoding and disable all BOM sniffing. This will
    /// always result in searching the raw bytes, regardless of their
    /// true encoding.
    Disabled,
}

impl EncodingMode {
    /// Checks if an explicit encoding has been set. Returns false for
    /// automatic BOM sniffing and no sniffing.
    ///
    /// This is only used to determine whether PCRE2 needs to have its own
    /// UTF-8 checking enabled. If we have an explicit encoding set, then
    /// we're always guaranteed to get UTF-8, so we can disable PCRE2's check.
    /// Otherwise, we have no such guarantee, and must enable PCRE2' UTF-8
    /// check.
    #[cfg(feature = "pcre2")]
    fn has_explicit_encoding(&self) -> bool {
        match self {
            EncodingMode::Some(_) => true,
            _ => false,
        }
    }
}

impl ArgMatches {
    /// Create an ArgMatches from clap's parse result.
    fn new(clap_matches: clap::ArgMatches<'static>) -> ArgMatches {
        ArgMatches(clap_matches)
    }

    /// Run clap and return the matches using a config file if present. If clap
    /// determines a problem with the user provided arguments (or if --help or
    /// --version are given), then an error/usage/version will be printed and
    /// the process will exit.
    ///
    /// If there are no additional arguments from the environment (e.g., a
    /// config file), then the given matches are returned as is.
    fn reconfigure(self) -> Result<ArgMatches> {
        // If the end user says no config, then respect it.
        if self.is_present("no-config") {
            log::debug!("not reading config files because --no-config is present");
            return Ok(self);
        }
        // If the user wants ripgrep to use a config file, then parse args
        // from that first.
        let mut args = config::args();
        if args.is_empty() {
            return Ok(self);
        }
        let mut cliargs = env::args_os();
        if let Some(bin) = cliargs.next() {
            args.insert(0, bin);
        }
        args.extend(cliargs);
        log::debug!("final argv: {:?}", args);
        Ok(ArgMatches(clap_matches(args)?))
    }

    /// Convert the result of parsing CLI arguments into ripgrep's higher level
    /// configuration structure.
    fn to_args(self) -> Result<Args> {
        // We compute these once since they could be large.
        let patterns = self.patterns()?;
        let matcher = self.matcher(&patterns)?;
        let mut paths = self.paths();
        let using_default_path = if paths.is_empty() {
            paths.push(self.path_default());
            true
        } else {
            false
        };
        Ok(Args(Arc::new(ArgsImp {
            matches: self,
            patterns,
            matcher,
            paths,
            using_default_path,
        })))
    }
}

/// High level routines for converting command line arguments into various
/// data structures used by ripgrep.
///
/// Methods are sorted alphabetically.
impl ArgMatches {
    /// Return the matcher that should be used for searching.
    ///
    /// If there was a problem building the matcher (e.g., a syntax error),
    /// then this returns an error.
    fn matcher(&self, patterns: &[String]) -> Result<PatternMatcher> {
        if self.is_present("pcre2") {
            self.matcher_engine("pcre2", patterns)
        } else {
            let engine = self.value_of_lossy("engine").unwrap();
            self.matcher_engine(&engine, patterns)
        }
    }

    /// Return the matcher that should be used for searching using engine
    /// as the engine for the patterns.
    ///
    /// If there was a problem building the matcher (e.g., a syntax error),
    /// then this returns an error.
    fn matcher_engine(&self, engine: &str, patterns: &[String]) -> Result<PatternMatcher> {
        match engine {
            "default" => {
                let matcher = match self.matcher_rust(patterns) {
                    Ok(matcher) => matcher,
                    Err(err) => {
                        return Err(From::from(suggest(err.to_string())));
                    }
                };
                Ok(PatternMatcher::RustRegex(matcher))
            }
            #[cfg(feature = "pcre2")]
            "pcre2" => {
                let matcher = self.matcher_pcre2(patterns)?;
                Ok(PatternMatcher::PCRE2(matcher))
            }
            #[cfg(not(feature = "pcre2"))]
            "pcre2" => Err(From::from(
                "PCRE2 is not available in this build of ripgrep",
            )),
            "auto" => {
                let rust_err = match self.matcher_rust(patterns) {
                    Ok(matcher) => {
                        return Ok(PatternMatcher::RustRegex(matcher));
                    }
                    Err(err) => err,
                };
                log::debug!("error building Rust regex in hybrid mode:\n{}", rust_err,);

                let pcre_err = match self.matcher_engine("pcre2", patterns) {
                    Ok(matcher) => return Ok(matcher),
                    Err(err) => err,
                };
                Err(From::from(format!(
                    "regex could not be compiled with either the default \
                     regex engine or with PCRE2.\n\n\
                     default regex engine error:\n{}\n{}\n{}\n\n\
                     PCRE2 regex engine error:\n{}",
                    "~".repeat(79),
                    rust_err,
                    "~".repeat(79),
                    pcre_err,
                )))
            }
            _ => Err(From::from(format!(
                "unrecognized regex engine '{}'",
                engine
            ))),
        }
    }

    /// Build a matcher using Rust's regex engine.
    ///
    /// If there was a problem building the matcher (such as a regex syntax
    /// error), then an error is returned.
    fn matcher_rust(&self, patterns: &[String]) -> Result<RustRegexMatcher> {
        let mut builder = RustRegexMatcherBuilder::new();
        builder
            .case_smart(self.case_smart())
            .case_insensitive(self.case_insensitive())
            .multi_line(true)
            .unicode(self.unicode())
            .octal(false)
            .word(self.is_present("word-regexp"));
        builder
            .line_terminator(Some(b'\n'))
            .dot_matches_new_line(false);
        if self.is_present("crlf") {
            builder.crlf(true);
        }
        if self.is_present("null-data") {
            builder.line_terminator(Some(b'\x00'));
        }
        if let Some(limit) = self.regex_size_limit()? {
            builder.size_limit(limit);
        }
        if let Some(limit) = self.dfa_size_limit()? {
            builder.dfa_size_limit(limit);
        }
        let res = if self.is_present("fixed-strings") {
            builder.build_literals(patterns)
        } else {
            builder.build(&patterns.join("|"))
        };
        Ok(res?)
    }

    /// Build a matcher using PCRE2.
    ///
    /// If there was a problem building the matcher (such as a regex syntax
    /// error), then an error is returned.
    #[cfg(feature = "pcre2")]
    fn matcher_pcre2(&self, patterns: &[String]) -> Result<PCRE2RegexMatcher> {
        let mut builder = PCRE2RegexMatcherBuilder::new();
        builder
            .case_smart(self.case_smart())
            .caseless(self.case_insensitive())
            .multi_line(true)
            .word(self.is_present("word-regexp"));
        // For whatever reason, the JIT craps out during regex compilation with
        // a "no more memory" error on 32 bit systems. So don't use it there.
        if cfg!(target_pointer_width = "64") {
            builder
                .jit_if_available(true)
                // The PCRE2 docs say that 32KB is the default, and that 1MB
                // should be big enough for anything. But let's crank it to
                // 10MB.
                .max_jit_stack_size(Some(10 * (1 << 20)));
        }
        if self.unicode() {
            builder.utf(true).ucp(true);
            if self.encoding()?.has_explicit_encoding() {
                // SAFETY: If an encoding was specified, then we're guaranteed
                // to get valid UTF-8, so we can disable PCRE2's UTF checking.
                // (Feeding invalid UTF-8 to PCRE2 is undefined behavior.)
                unsafe {
                    builder.disable_utf_check();
                }
            }
        }
        if self.is_present("multiline") {
            builder.dotall(self.is_present("multiline-dotall"));
        }
        if self.is_present("crlf") {
            builder.crlf(true);
        }
        Ok(builder.build(&patterns.join("|"))?)
    }

    /// Build a Patch printer that writes results to the given writer.
    fn printer_patch<W: io::Write>(&self, wtr: W) -> Result<Patch<W>> {
        let mut builder = PatchBuilder::new();
        builder.replacement(self.replacement()?);
        Ok(builder.build(wtr))
    }

    /// Build a searcher from the command line parameters.
    fn searcher(&self, paths: &[PathBuf]) -> Result<Searcher> {
        let (ctx_before, ctx_after) = self.contexts()?;
        let line_term = if self.is_present("crlf") {
            LineTerminator::crlf()
        } else if self.is_present("null-data") {
            LineTerminator::byte(b'\x00')
        } else {
            LineTerminator::byte(b'\n')
        };
        let mut builder = SearcherBuilder::new();
        builder
            .line_terminator(line_term)
            .invert_match(self.is_present("invert-match"))
            .line_number(true)
            .multi_line(self.is_present("multiline"))
            .before_context(ctx_before)
            .after_context(ctx_after)
            .passthru(self.is_present("passthru"))
            .memory_map(self.mmap_choice(paths));
        match self.encoding()? {
            EncodingMode::Some(enc) => {
                builder.encoding(Some(enc));
            }
            EncodingMode::Auto => {} // default for the searcher
            EncodingMode::Disabled => {
                builder.bom_sniffing(false);
            }
        }
        Ok(builder.build())
    }

    /// Return a builder for recursively traversing a directory while
    /// respecting ignore rules.
    ///
    /// If there was a problem parsing the CLI arguments necessary for
    /// constructing the builder, then this returns an error.
    fn walker_builder(&self, paths: &[PathBuf]) -> Result<WalkBuilder> {
        let mut builder = WalkBuilder::new(&paths[0]);
        for path in &paths[1..] {
            builder.add(path);
        }
        if !self.no_ignore_files() {
            for path in self.ignore_paths() {
                if let Some(err) = builder.add_ignore(path) {
                    ignore_message!("{}", err);
                }
            }
        }
        builder
            .max_depth(self.usize_of("max-depth")?)
            .follow_links(self.is_present("follow"))
            .max_filesize(self.max_file_size()?)
            .threads(self.threads()?)
            .same_file_system(self.is_present("one-file-system"))
            .skip_stdout(!self.is_present("files"))
            .overrides(self.overrides()?)
            .types(self.types()?)
            .hidden(!self.hidden())
            .parents(!self.no_ignore_parent())
            .ignore(!self.no_ignore_dot())
            .git_global(!self.no_ignore_vcs() && !self.no_ignore_global())
            .git_ignore(!self.no_ignore_vcs())
            .git_exclude(!self.no_ignore_vcs() && !self.no_ignore_exclude())
            .require_git(!self.is_present("no-require-git"))
            .ignore_case_insensitive(self.ignore_file_case_insensitive());
        if !self.no_ignore() {
            builder.add_custom_ignore_filename(".rgignore");
        }
        Ok(builder)
    }
}

/// Mid level routines for converting command line arguments into various types
/// of data structures.
///
/// Methods are sorted alphabetically.
impl ArgMatches {
    /// Returns the form of binary detection to perform on files that are
    /// implicitly searched via recursive directory traversal.
    fn binary_detection_implicit(&self) -> BinaryDetection {
        let none = self.is_present("text") || self.is_present("null-data");
        let convert = self.is_present("binary") || self.unrestricted_count() >= 3;
        if none {
            BinaryDetection::none()
        } else if convert {
            BinaryDetection::convert(b'\x00')
        } else {
            BinaryDetection::quit(b'\x00')
        }
    }

    /// Returns the form of binary detection to perform on files that are
    /// explicitly searched via the user invoking ripgrep on a particular
    /// file or files or stdin.
    ///
    /// In general, this should never be BinaryDetection::quit, since that acts
    /// as a filter (but quitting immediately once a NUL byte is seen), and we
    /// should never filter out files that the user wants to explicitly search.
    fn binary_detection_explicit(&self) -> BinaryDetection {
        let none = self.is_present("text") || self.is_present("null-data");
        if none {
            BinaryDetection::none()
        } else {
            BinaryDetection::convert(b'\x00')
        }
    }

    /// Returns true if the command line configuration implies that a match
    /// can never be shown.
    fn can_never_match(&self, patterns: &[String]) -> bool {
        patterns.is_empty()
    }

    /// Returns true if and only if case should be ignored.
    ///
    /// If --case-sensitive is present, then case is never ignored, even if
    /// --ignore-case is present.
    fn case_insensitive(&self) -> bool {
        self.is_present("ignore-case") && !self.is_present("case-sensitive")
    }

    /// Returns true if and only if smart case has been enabled.
    ///
    /// If either --ignore-case of --case-sensitive are present, then smart
    /// case is disabled.
    fn case_smart(&self) -> bool {
        self.is_present("smart-case")
            && !self.is_present("ignore-case")
            && !self.is_present("case-sensitive")
    }

    /// Returns the before and after contexts from the command line.
    ///
    /// RipPatch *always* uses a context of 3.
    fn contexts(&self) -> Result<(usize, usize)> {
        let after = self.usize_of("after-context")?.unwrap_or(0);
        let before = self.usize_of("before-context")?.unwrap_or(0);
        let both = self.usize_of("context")?.unwrap_or(0);
        Ok(if both > 0 {
            (both, both)
        } else {
            (before, after)
        })
    }

    /// Parse the dfa-size-limit argument option into a byte count.
    fn dfa_size_limit(&self) -> Result<Option<usize>> {
        let r = self.parse_human_readable_size("dfa-size-limit")?;
        u64_to_usize("dfa-size-limit", r)
    }

    /// Returns the encoding mode to use.
    ///
    /// This only returns an encoding if one is explicitly specified. Otherwise
    /// if set to automatic, the Searcher will do BOM sniffing for UTF-16
    /// and transcode seamlessly. If disabled, no BOM sniffing nor transcoding
    /// will occur.
    fn encoding(&self) -> Result<EncodingMode> {
        if self.is_present("no-encoding") {
            return Ok(EncodingMode::Auto);
        }

        let label = match self.value_of_lossy("encoding") {
            None if self.pcre2_unicode() => "utf-8".to_string(),
            None => return Ok(EncodingMode::Auto),
            Some(label) => label,
        };

        if label == "auto" {
            return Ok(EncodingMode::Auto);
        } else if label == "none" {
            return Ok(EncodingMode::Disabled);
        }

        Ok(EncodingMode::Some(Encoding::new(&label)?))
    }

    /// Returns true if and only if hidden files/directories should be
    /// searched.
    fn hidden(&self) -> bool {
        self.is_present("hidden") || self.unrestricted_count() >= 2
    }

    /// Returns true if ignore files should be processed case insensitively.
    fn ignore_file_case_insensitive(&self) -> bool {
        self.is_present("ignore-file-case-insensitive")
    }

    /// Return all of the ignore file paths given on the command line.
    fn ignore_paths(&self) -> Vec<PathBuf> {
        let paths = match self.values_of_os("ignore-file") {
            None => return vec![],
            Some(paths) => paths,
        };
        paths.map(|p| Path::new(p).to_path_buf()).collect()
    }

    /// Parses the max-filesize argument option into a byte count.
    fn max_file_size(&self) -> Result<Option<u64>> {
        self.parse_human_readable_size("max-filesize")
    }

    /// Returns whether we should attempt to use memory maps or not.
    fn mmap_choice(&self, paths: &[PathBuf]) -> MmapChoice {
        // SAFETY: Memory maps are difficult to impossible to encapsulate
        // safely in a portable way that doesn't simultaneously negate some of
        // the benfits of using memory maps. For ripgrep's use, we never mutate
        // a memory map and generally never store the contents of memory map
        // in a data structure that depends on immutability. Generally
        // speaking, the worst thing that can happen is a SIGBUS (if the
        // underlying file is truncated while reading it), which will cause
        // ripgrep to abort. This reasoning should be treated as suspect.
        let maybe = unsafe { MmapChoice::auto() };
        let never = MmapChoice::never();
        if self.is_present("no-mmap") {
            never
        } else if self.is_present("mmap") {
            maybe
        } else if paths.len() <= 10 && paths.iter().all(|p| p.is_file()) {
            // If we're only searching a few paths and all of them are
            // files, then memory maps are probably faster.
            maybe
        } else {
            never
        }
    }

    /// Returns true if ignore files should be ignored.
    fn no_ignore(&self) -> bool {
        self.is_present("no-ignore") || self.unrestricted_count() >= 1
    }

    /// Returns true if .ignore files should be ignored.
    fn no_ignore_dot(&self) -> bool {
        self.is_present("no-ignore-dot") || self.no_ignore()
    }

    /// Returns true if local exclude (ignore) files should be ignored.
    fn no_ignore_exclude(&self) -> bool {
        self.is_present("no-ignore-exclude") || self.no_ignore()
    }

    /// Returns true if explicitly given ignore files should be ignored.
    fn no_ignore_files(&self) -> bool {
        // We don't look at no-ignore here because --no-ignore is explicitly
        // documented to not override --ignore-file. We could change this, but
        // it would be a fairly severe breaking change.
        self.is_present("no-ignore-files")
    }

    /// Returns true if global ignore files should be ignored.
    fn no_ignore_global(&self) -> bool {
        self.is_present("no-ignore-global") || self.no_ignore()
    }

    /// Returns true if parent ignore files should be ignored.
    fn no_ignore_parent(&self) -> bool {
        self.is_present("no-ignore-parent") || self.no_ignore()
    }

    /// Returns true if VCS ignore files should be ignored.
    fn no_ignore_vcs(&self) -> bool {
        self.is_present("no-ignore-vcs") || self.no_ignore()
    }

    /// Builds the set of glob overrides from the command line flags.
    fn overrides(&self) -> Result<Override> {
        let globs = self.values_of_lossy_vec("glob");
        let iglobs = self.values_of_lossy_vec("iglob");
        if globs.is_empty() && iglobs.is_empty() {
            return Ok(Override::empty());
        }

        let mut builder = OverrideBuilder::new(current_dir()?);
        // Make all globs case insensitive with --glob-case-insensitive.
        if self.is_present("glob-case-insensitive") {
            builder.case_insensitive(true).unwrap();
        }
        for glob in globs {
            builder.add(&glob)?;
        }
        // This only enables case insensitivity for subsequent globs.
        builder.case_insensitive(true).unwrap();
        for glob in iglobs {
            builder.add(&glob)?;
        }
        Ok(builder.build()?)
    }

    /// Return all file paths that ripgrep should search.
    ///
    /// If no paths were given, then this returns an empty list.
    fn paths(&self) -> Vec<PathBuf> {
        let mut paths: Vec<PathBuf> = match self.values_of_os("path") {
            None => vec![],
            Some(paths) => paths.map(|p| Path::new(p).to_path_buf()).collect(),
        };
        // If --file, --files or --regexp is given, then the first path is
        // always in `pattern`.
        if self.is_present("file") || self.is_present("files") || self.is_present("regexp") {
            if let Some(path) = self.value_of_os("pattern") {
                paths.insert(0, Path::new(path).to_path_buf());
            }
        }
        paths
    }

    /// Return the default path that ripgrep should search. This should only
    /// be used when ripgrep is not otherwise given at least one file path
    /// as a positional argument.
    fn path_default(&self) -> PathBuf {
        let file_is_stdin = self
            .values_of_os("file")
            .map_or(false, |mut files| files.any(|f| f == "-"));
        let search_cwd = !cli::is_readable_stdin()
            || (self.is_present("file") && file_is_stdin)
            || self.is_present("files")
            || self.is_present("type-list")
            || self.is_present("pcre2-version");
        if search_cwd {
            Path::new("./").to_path_buf()
        } else {
            Path::new("-").to_path_buf()
        }
    }

    /// Get a sequence of all available patterns from the command line.
    /// This includes reading the -e/--regexp and -f/--file flags.
    ///
    /// Note that if -F/--fixed-strings is set, then all patterns will be
    /// escaped. If -x/--line-regexp is set, then all patterns are surrounded
    /// by `^...$`. Other things, such as --word-regexp, are handled by the
    /// regex matcher itself.
    ///
    /// If any pattern is invalid UTF-8, then an error is returned.
    fn patterns(&self) -> Result<Vec<String>> {
        if self.is_present("files") || self.is_present("type-list") {
            return Ok(vec![]);
        }
        let mut pats = vec![];
        match self.values_of_os("regexp") {
            None => {
                if self.values_of_os("file").is_none() {
                    if let Some(os_pat) = self.value_of_os("pattern") {
                        pats.push(self.pattern_from_os_str(os_pat)?);
                    }
                }
            }
            Some(os_pats) => {
                for os_pat in os_pats {
                    pats.push(self.pattern_from_os_str(os_pat)?);
                }
            }
        }
        if let Some(paths) = self.values_of_os("file") {
            for path in paths {
                if path == "-" {
                    pats.extend(
                        cli::patterns_from_stdin()?
                            .into_iter()
                            .map(|p| self.pattern_from_string(p)),
                    );
                } else {
                    pats.extend(
                        cli::patterns_from_path(path)?
                            .into_iter()
                            .map(|p| self.pattern_from_string(p)),
                    );
                }
            }
        }
        Ok(pats)
    }

    /// Returns a pattern that is guaranteed to produce an empty regular
    /// expression that is valid in any position.
    fn pattern_empty(&self) -> String {
        // This would normally just be an empty string, which works on its
        // own, but if the patterns are joined in a set of alternations, then
        // you wind up with `foo|`, which is currently invalid in Rust's regex
        // engine.
        "(?:z{0})*".to_string()
    }

    /// Converts an OsStr pattern to a String pattern. The pattern is escaped
    /// if -F/--fixed-strings is set.
    ///
    /// If the pattern is not valid UTF-8, then an error is returned.
    fn pattern_from_os_str(&self, pat: &OsStr) -> Result<String> {
        let s = cli::pattern_from_os(pat)?;
        Ok(self.pattern_from_str(s))
    }

    /// Converts a &str pattern to a String pattern. The pattern is escaped
    /// if -F/--fixed-strings is set.
    fn pattern_from_str(&self, pat: &str) -> String {
        self.pattern_from_string(pat.to_string())
    }

    /// Applies additional processing on the given pattern if necessary
    /// (such as escaping meta characters or turning it into a line regex).
    fn pattern_from_string(&self, pat: String) -> String {
        let pat = self.pattern_line(self.pattern_literal(pat));
        if pat.is_empty() {
            self.pattern_empty()
        } else {
            pat
        }
    }

    /// Returns the given pattern as a line pattern if the -x/--line-regexp
    /// flag is set. Otherwise, the pattern is returned unchanged.
    fn pattern_line(&self, pat: String) -> String {
        if self.is_present("line-regexp") {
            format!(r"^(?:{})$", pat)
        } else {
            pat
        }
    }

    /// Returns the given pattern as a literal pattern if the
    /// -F/--fixed-strings flag is set. Otherwise, the pattern is returned
    /// unchanged.
    fn pattern_literal(&self, pat: String) -> String {
        if self.is_present("fixed-strings") {
            regex::escape(&pat)
        } else {
            pat
        }
    }

    /// Returns the preprocessor command if one was specified.
    fn preprocessor(&self) -> Option<PathBuf> {
        let path = match self.value_of_os("pre") {
            None => return None,
            Some(path) => path,
        };
        if path.is_empty() {
            return None;
        }
        Some(Path::new(path).to_path_buf())
    }

    /// Builds the set of globs for filtering files to apply to the --pre
    /// flag. If no --pre-globs are available, then this always returns an
    /// empty set of globs.
    fn preprocessor_globs(&self) -> Result<Override> {
        let globs = self.values_of_lossy_vec("pre-glob");
        if globs.is_empty() {
            return Ok(Override::empty());
        }
        let mut builder = OverrideBuilder::new(current_dir()?);
        for glob in globs {
            builder.add(&glob)?;
        }
        Ok(builder.build()?)
    }

    /// Parse the regex-size-limit argument option into a byte count.
    fn regex_size_limit(&self) -> Result<Option<usize>> {
        let r = self.parse_human_readable_size("regex-size-limit")?;
        u64_to_usize("regex-size-limit", r)
    }

    /// Returns the replacement string as UTF-8 bytes; it must exist.
    fn replacement(&self) -> Result<Vec<u8>> {
        let r = self
            .value_of_lossy("pos_replace")
            .or(self.value_of_lossy("replace"))
            .map(|s| s.into_bytes())
            .ok_or(ERR_NO_REPLACEMENT)?;
        Ok(r)
    }

    /// Return the number of threads that should be used for parallelism.
    fn threads(&self) -> Result<usize> {
        let threads = self.usize_of("threads")?.unwrap_or(0);
        Ok(if threads == 0 {
            cmp::min(12, num_cpus::get())
        } else {
            threads
        })
    }

    /// Builds a file type matcher from the command line flags.
    fn types(&self) -> Result<Types> {
        let mut builder = TypesBuilder::new();
        builder.add_defaults();
        for ty in self.values_of_lossy_vec("type-clear") {
            builder.clear(&ty);
        }
        for def in self.values_of_lossy_vec("type-add") {
            builder.add_def(&def)?;
        }
        for ty in self.values_of_lossy_vec("type") {
            builder.select(&ty);
        }
        for ty in self.values_of_lossy_vec("type-not") {
            builder.negate(&ty);
        }
        builder.build().map_err(From::from)
    }

    /// Returns the number of times the `unrestricted` flag is provided.
    fn unrestricted_count(&self) -> u64 {
        self.occurrences_of("unrestricted")
    }

    /// Returns true if and only if Unicode mode should be enabled.
    fn unicode(&self) -> bool {
        // Unicode mode is enabled by default, so only disable it when
        // --no-unicode is given explicitly.
        !(self.is_present("no-unicode") || self.is_present("no-pcre2-unicode"))
    }

    /// Returns true if and only if PCRE2 is enabled and its Unicode mode is
    /// enabled.
    fn pcre2_unicode(&self) -> bool {
        self.is_present("pcre2") && self.unicode()
    }
}

/// Lower level generic helper methods for teasing values out of clap.
impl ArgMatches {
    /// Like values_of_lossy, but returns an empty vec if the flag is not
    /// present.
    fn values_of_lossy_vec(&self, name: &str) -> Vec<String> {
        self.values_of_lossy(name).unwrap_or_else(Vec::new)
    }

    /// Safely reads an arg value with the given name, and if it's present,
    /// tries to parse it as a usize value.
    ///
    /// If the number is zero, then it is considered absent and `None` is
    /// returned.
    fn usize_of_nonzero(&self, name: &str) -> Result<Option<usize>> {
        let n = match self.usize_of(name)? {
            None => return Ok(None),
            Some(n) => n,
        };
        Ok(if n == 0 { None } else { Some(n) })
    }

    /// Safely reads an arg value with the given name, and if it's present,
    /// tries to parse it as a usize value.
    fn usize_of(&self, name: &str) -> Result<Option<usize>> {
        match self.value_of_lossy(name) {
            None => Ok(None),
            Some(v) => v.parse().map(Some).map_err(From::from),
        }
    }

    /// Parses an argument of the form `[0-9]+(KMG)?`.
    ///
    /// If the aforementioned format is not recognized, then this returns an
    /// error.
    fn parse_human_readable_size(&self, arg_name: &str) -> Result<Option<u64>> {
        let size = match self.value_of_lossy(arg_name) {
            None => return Ok(None),
            Some(size) => size,
        };
        Ok(Some(cli::parse_human_readable_size(&size)?))
    }
}

/// The following methods mostly dispatch to the underlying clap methods
/// directly. Methods that would otherwise get a single value will fetch all
/// values and return the last one. (Clap returns the first one.) We only
/// define the ones we need.
impl ArgMatches {
    fn is_present(&self, name: &str) -> bool {
        self.0.is_present(name)
    }

    fn occurrences_of(&self, name: &str) -> u64 {
        self.0.occurrences_of(name)
    }

    fn value_of_lossy(&self, name: &str) -> Option<String> {
        self.0.value_of_lossy(name).map(|s| s.into_owned())
    }

    fn values_of_lossy(&self, name: &str) -> Option<Vec<String>> {
        self.0.values_of_lossy(name)
    }

    fn value_of_os(&self, name: &str) -> Option<&OsStr> {
        self.0.value_of_os(name)
    }

    fn values_of_os(&self, name: &str) -> Option<clap::OsValues<'_>> {
        self.0.values_of_os(name)
    }
}

/// Inspect an error resulting from building a Rust regex matcher, and if it's
/// believed to correspond to a syntax error that another engine could handle,
/// then add a message to suggest the use of the engine flag.
fn suggest(msg: String) -> String {
    if let Some(pcre_msg) = suggest_pcre2(&msg) {
        return pcre_msg;
    }
    msg
}

/// Inspect an error resulting from building a Rust regex matcher, and if it's
/// believed to correspond to a syntax error that PCRE2 could handle, then
/// add a message to suggest the use of -P/--pcre2.
fn suggest_pcre2(msg: &str) -> Option<String> {
    #[cfg(feature = "pcre2")]
    fn suggest(msg: &str) -> Option<String> {
        if !msg.contains("backreferences") && !msg.contains("look-around") {
            None
        } else {
            Some(format!(
                "{}

Consider enabling PCRE2 with the --pcre2 flag, which can handle backreferences
and look-around.",
                msg
            ))
        }
    }

    #[cfg(not(feature = "pcre2"))]
    fn suggest(_: &str) -> Option<String> {
        None
    }

    suggest(msg)
}

/// Convert the result of parsing a human readable file size to a `usize`,
/// failing if the type does not fit.
fn u64_to_usize(arg_name: &str, value: Option<u64>) -> Result<Option<usize>> {
    use std::usize;

    let value = match value {
        None => return Ok(None),
        Some(value) => value,
    };
    if value <= usize::MAX as u64 {
        Ok(Some(value as usize))
    } else {
        Err(From::from(format!("number too large for {}", arg_name)))
    }
}

/// Returns a clap matches object if the given arguments parse successfully.
///
/// Otherwise, if an error occurred, then it is returned unless the error
/// corresponds to a `--help` or `--version` request. In which case, the
/// corresponding output is printed and the current process is exited
/// successfully.
fn clap_matches<I, T>(args: I) -> Result<clap::ArgMatches<'static>>
where
    I: IntoIterator<Item = T>,
    T: Into<OsString> + Clone,
{
    let err = match app::app().get_matches_from_safe(args) {
        Ok(matches) => return Ok(matches),
        Err(err) => err,
    };
    if err.use_stderr() {
        return Err(err.into());
    }
    // Explicitly ignore any error returned by write!. The most likely error
    // at this point is a broken pipe error, in which case, we want to ignore
    // it and exit quietly.
    //
    // (This is the point of this helper function. clap's functionality for
    // doing this will panic on a broken pipe error.)
    let _ = write!(io::stdout(), "{}", err);
    process::exit(0);
}

/// Attempts to discover the current working directory. This mostly just defers
/// to the standard library, however, such things will fail if ripgrep is in
/// a directory that no longer exists. We attempt some fallback mechanisms,
/// such as querying the PWD environment variable, but otherwise return an
/// error.
fn current_dir() -> Result<PathBuf> {
    let err = match env::current_dir() {
        Err(err) => err,
        Ok(cwd) => return Ok(cwd),
    };
    if let Some(cwd) = env::var_os("PWD") {
        if !cwd.is_empty() {
            return Ok(PathBuf::from(cwd));
        }
    }
    Err(format!(
        "failed to get current working directory: {} \
         --- did your CWD get deleted?",
        err,
    )
    .into())
}
