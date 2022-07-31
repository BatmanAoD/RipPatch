use std::process;

mod patch;
mod patcht;
mod util;

fn main() {
    /*
    if let Err(err) = Args::parse().and_then(search_parallel) {
        eprintln!("{}", err);
        process::exit(2);
    }
    */
} 

// XXX copied from rg
/*
fn search_parallel(args: &Args) -> Result<bool> {
    use std::sync::atomic::AtomicBool;
    use std::sync::atomic::Ordering::SeqCst;

    let quit_after_match = args.quit_after_match()?;
    let started_at = Instant::now();
    let subject_builder = args.subject_builder();
    let bufwtr = args.buffer_writer()?;
    let stats = args.stats()?.map(Mutex::new);
    let matched = AtomicBool::new(false);
    let searched = AtomicBool::new(false);
    let mut searcher_err = None;
    args.walker_parallel()?.run(|| {
        let bufwtr = &bufwtr;
        let stats = &stats;
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
            if let Some(ref locked_stats) = *stats {
                let mut stats = locked_stats.lock().unwrap();
                *stats += search_result.stats().unwrap();
            }
            if let Err(err) = bufwtr.print(searcher.printer().get_mut()) {
                // A broken pipe means graceful termination.
                if err.kind() == io::ErrorKind::BrokenPipe {
                    return WalkState::Quit;
                }
                // Otherwise, we continue on our merry way.
                err_message!("{}: {}", subject.path().display(), err);
            }
            if matched.load(SeqCst) && quit_after_match {
                WalkState::Quit
            } else {
                WalkState::Continue
            }
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
*/