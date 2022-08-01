use std::io;

use grep::matcher::{Match, Matcher};
use grep::searcher::{Searcher, SinkError};

// Taken from grep::printer::util directly, since that version is not public;
// however, the multiline-search case is removed.
pub fn find_iter_at_in_context<M, F>(
    searcher: &Searcher,
    matcher: M,
    mut bytes: &[u8],
    range: std::ops::Range<usize>,
    mut matched: F,
) -> io::Result<()>
where
    M: Matcher,
    F: FnMut(Match) -> bool,
{
    // Multi-line search is not supported.
    // When searching a single line, we should remove the line terminator.
    // Otherwise, it's possible for the regex (via look-around) to observe
    // the line terminator and not match because of it.
    let mut m = Match::new(0, range.end);
    trim_line_terminator(searcher, bytes, &mut m);
    bytes = &bytes[..m.end()];
    matcher
        .find_iter_at(bytes, range.start, |m| {
            if m.start() >= range.end {
                return false;
            }
            matched(m)
        })
        .map_err(io::Error::error_message)
}

/// Given a buf and some bounds, if there is a line terminator at the end of
/// the given bounds in buf, then the bounds are trimmed to remove the line
/// terminator.
pub fn trim_line_terminator(searcher: &Searcher, buf: &[u8], line: &mut Match) {
    let lineterm = searcher.line_terminator();
    if lineterm.is_suffix(&buf[*line]) {
        let mut end = line.end() - 1;
        if lineterm.is_crlf() && end > 0 && buf.get(end - 1) == Some(&b'\r') {
            end -= 1;
        }
        *line = line.with_end(end);
    }
}
