use std::error;
use std::io;
use std::process;

use ignore::WalkState;

use args::Args;

#[macro_use]
mod messages;

mod app;
mod args;
mod config;
mod logger;
mod patch;
mod patcht;
mod search;
mod subject;
mod util;

type Result<T> = ::std::result::Result<T, Box<dyn error::Error>>;

fn main() {
    if let Err(err) = Args::parse().and_then(search_parallel) {
        eprintln!("{}", err);
        process::exit(2);
    }
}

fn search_parallel(args: Args) -> Result<bool> {
    use std::sync::atomic::AtomicBool;
    use std::sync::atomic::Ordering::SeqCst;

    let subject_builder = args.subject_builder();
    let bufwtr = args.buffer_writer();
    let matched = AtomicBool::new(false);
    let searched = AtomicBool::new(false);
    let mut searcher_err = None;
    args.walker_parallel()?.run(|| {
        let bufwtr = &bufwtr;
        let matched = &matched;
        let searched = &searched;
        let subject_builder = &subject_builder;
        let mut searcher = match args.search_worker(bufwtr.buffer()) {
            Ok(searcher) => searcher,
            Err(err) => {
                searcher_err = Some(err);
                return Box::new(move |_| WalkState::Quit);
            }
        };

        Box::new(move |result| {
            let subject = match subject_builder.build_from_result(result) {
                Some(subject) => subject,
                None => return WalkState::Continue,
            };
            searched.store(true, SeqCst);
            searcher.printer().get_mut().clear();
            let search_result = match searcher.search(&subject) {
                Ok(search_result) => search_result,
                Err(err) => {
                    err_message!("{}: {}", subject.path().display(), err);
                    return WalkState::Continue;
                }
            };
            if search_result.has_match() {
                matched.store(true, SeqCst);
            }
            if let Err(err) = bufwtr.print(searcher.printer().get_mut()) {
                // A broken pipe means graceful termination.
                if err.kind() == io::ErrorKind::BrokenPipe {
                    return WalkState::Quit;
                }
                // Otherwise, we continue on our merry way.
                err_message!("{}: {}", subject.path().display(), err);
            }
            WalkState::Continue
        })
    });
    if let Some(err) = searcher_err.take() {
        return Err(err);
    }
    if args.using_default_path() && !searched.load(SeqCst) {
        eprint_nothing_searched();
    }
    Ok(matched.load(SeqCst))
}

fn eprint_nothing_searched() {
    // The application name is not specified, because the
    // guts of RipPatch *are* ripgrep.
    err_message!(
        "No files were searched, which means an unexpected filter
         may have been applied.\n\
         Running with --debug will show why files are being skipped."
    );
}
