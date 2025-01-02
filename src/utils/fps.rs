use std::time::{Instant, Duration};

pub struct FPSCounter {
    last_instant: Instant,
    frame_count: u32,
}

impl FPSCounter {
    pub fn new() -> Self {
        FPSCounter {
            last_instant: Instant::now(),
            frame_count: 0,
        }
    }

    /// Updates the FPS counter. Returns `Some(fps)` once per second.
    pub fn update(&mut self) -> Option<u32> {
        self.frame_count += 1;
        let now = Instant::now();
        let duration = now.duration_since(self.last_instant);

        if duration >= Duration::from_secs(1) {
            let fps = self.frame_count;
            self.frame_count = 0;
            self.last_instant = now;
            Some(fps)
        } else {
            None
        }
    }
}