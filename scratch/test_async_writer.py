import time
import queue
import threading
import cv2
import numpy as np

class AsyncVideoWriter:
    def __init__(self, filename: str, fourcc: int, fps: float, frame_size: tuple, max_queue: int = 128):
        self.writer = cv2.VideoWriter(str(filename), fourcc, fps, frame_size)
        self.queue = queue.Queue(maxsize=max_queue)
        self.stopped = False
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def isOpened(self):
        return self.writer.isOpened()

    def _worker(self):
        while True:
            frame = self.queue.get()
            if frame is None:
                self.queue.task_done()
                break
            self.writer.write(frame)
            self.queue.task_done()

    def write(self, frame: np.ndarray):
        if not self.stopped:
            self.queue.put(frame.copy())

    def release(self):
        if not self.stopped:
            self.stopped = True
            self.queue.put(None)
            self.thread.join()
            self.writer.release()

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)

# 1. Sync Writer
sync_writer1 = cv2.VideoWriter("scratch/sync1.mp4", fourcc, 25.0, (1280, 720))
sync_writer2 = cv2.VideoWriter("scratch/sync2.mp4", fourcc, 25.0, (1280, 720))

t0 = time.perf_counter()
for _ in range(50):
    sync_writer1.write(dummy_frame)
    sync_writer2.write(dummy_frame)
t_sync = time.perf_counter() - t0
sync_writer1.release()
sync_writer2.release()

# 2. Async Writer
async_writer1 = AsyncVideoWriter("scratch/async1.mp4", fourcc, 25.0, (1280, 720))
async_writer2 = AsyncVideoWriter("scratch/async2.mp4", fourcc, 25.0, (1280, 720))

t0 = time.perf_counter()
for _ in range(50):
    async_writer1.write(dummy_frame)
    async_writer2.write(dummy_frame)
t_async_loop = time.perf_counter() - t0
async_writer1.release()
async_writer2.release()

print(f"Sync write (main loop): {t_sync*1000/50:.2f} ms per frame ({50/t_sync:.1f} FPS)")
print(f"Async write (main loop): {t_async_loop*1000/50:.2f} ms per frame ({50/t_async_loop:.1f} FPS) -> {t_sync/t_async_loop:.1f}x faster loop!")
