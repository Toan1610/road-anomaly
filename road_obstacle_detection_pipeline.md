# Pipeline phát hiện dị vật/lấn chiếm lòng đường từ camera hành trình

## 1. Mục tiêu

Phát hiện các vùng/vật thể không phải phương tiện nằm trên lòng đường có thể lưu thông, **không yêu cầu hệ thống phải biết tên hoặc class của vật thể**. Ví dụ: đống cát, gạch, đá, thùng hàng, túi, vật rơi hoặc vật thể chưa từng gặp.

Phương tiện giao thông cần được loại trừ, bao gồm cả phương tiện đang chạy và đang đỗ.

Đây là pipeline đề xuất dựa trên các mô hình pretrained. Trước khi triển khai thực tế, cần kiểm tra checkpoint, giấy phép, môi trường phụ thuộc và chất lượng trên video camera hành trình của hệ thống.

## 2. Pipeline tổng thể

```text
Video camera hành trình
        |
        v
Giải mã frame + chuẩn hóa màu/kích thước
        |
        +-------------------------------+
        |                               |
        v                               v
Perspective-Aware Road             SegFormer Cityscapes
Obstacle Detection                 (hoặc model phân đoạn tương đương)
        |                               |
        v                               +--> road/drivable-region mask
Obstacle score map                    +--> vehicle mask
        |                               |
        +---------------+---------------+
                        |
                        v
        Hợp nhất mask và loại trừ phương tiện
                        |
                        v
        Lọc vùng nhỏ/nhiễu + theo dõi qua frame
                        |
                        v
        Vùng nghi vấn lấn chiếm + cảnh báo + overlay/log
```

### Nhánh A — Phát hiện vật cản

**Model ưu tiên thử:** Perspective-Aware Road Obstacle Detection (EPFL).

- Mục tiêu của phương pháp là phát hiện vật cản tùy ý nằm trên mặt đường, không yêu cầu từng vật cản có class riêng.
- Phương pháp dùng dữ liệu vật cản tổng hợp và thông tin phối cảnh; “không cần phân loại class lúc inference” không có nghĩa là model chưa từng được huấn luyện.
- Dùng checkpoint pretrained do tác giả cung cấp nếu có và tương thích với môi trường. Không tự giả định tên checkpoint, lệnh chạy hoặc định dạng đầu ra trước khi kiểm tra README và thư mục `src` của repo.
- Đầu ra cần được chuyển thành `obstacle_score_map` có kích thước tương ứng với frame gốc, trong đó giá trị cao hơn thể hiện mức nghi vấn vật cản.

Repo: https://github.com/cvlab-epfl/perspective-aware-obstacles

### Nhánh B — Mặt đường và phương tiện

**Model baseline:** SegFormer pretrained trên Cityscapes; có thể thay bằng một model semantic segmentation tương đương nếu chất lượng thực tế tốt hơn.

Lấy các đầu ra:
- `road_mask` hoặc vùng đường có thể lưu thông.
- `vehicle_mask`: tối thiểu gồm car, truck, bus, train, motorcycle, bicycle. Điều chỉnh theo các class có trong checkpoint.
- Nếu chính sách cần loại trừ cả người đang cưỡi xe, cân nhắc thêm class `rider`.

Phải đọc mapping class từ cấu hình model, không hard-code ID lớp khi chưa xác nhận checkpoint. Mask phương tiện có thể sai, đặc biệt với xe đỗ bị che khuất hoặc nhìn từ góc bất thường.

Tài liệu: https://huggingface.co/docs/transformers/model_doc/segformer  
Model NVIDIA TAO tham khảo: https://catalog.ngc.nvidia.com/orgs/nvidia/tao/models/pretrained_segformer_cityscapes

## 3. Hợp nhất đầu ra

Sau khi đưa tất cả mask về cùng kích thước frame:

```python
candidate_mask = (
    (obstacle_score_map >= obstacle_threshold)
    & road_mask
    & (~vehicle_mask)
)
```

Ý nghĩa:
- `obstacle_score_map >= obstacle_threshold`: model vật cản đánh giá vùng đủ đáng nghi.
- `road_mask`: chỉ giữ vùng nằm trong vùng đường quan tâm.
- `~vehicle_mask`: loại bỏ pixel được nhận diện là phương tiện.

Đây là mã logic minh họa, không phải chương trình hoàn chỉnh. Tên biến và cách lấy score phải khớp API thật của checkpoint.

### Xác định vùng đường

Mask class `road` của Cityscapes là một baseline, nhưng có thể không khớp hoàn toàn với vùng có thể lưu thông trong mọi tuyến đường. Với camera hành trình, nên đánh giá riêng:
- đường đô thị, đường nông thôn và đường không có vạch;
- đường cong, dốc và giao lộ;
- xe đỗ che một phần mặt đường;
- cảnh mưa, đêm và ánh sáng gắt.

Không nên chỉ dùng phép đóng hình thái học (morphological closing) để tự suy ra vùng đường bị che: vật cản lớn có thể không được khôi phục đúng và phép đóng có thể lấp nhầm các vùng khác.

## 4. Hậu xử lý và cảnh báo

1. **Lọc nhiễu:** dùng connected components để gom pixel liền kề thành vùng; lọc theo diện tích tối thiểu và hình học phù hợp với độ phân giải.
2. **Không đặt một ngưỡng diện tích cố định cho mọi khoảng cách:** vật cản nhỏ ở xa có thể quan trọng. Có thể dùng ngưỡng theo vùng ảnh/phối cảnh sau khi kiểm thử.
3. **Theo dõi qua frame:** liên kết các vùng giữa frame bằng IoU mask, tâm vùng hoặc tracker phù hợp.
4. **Hai mức cảnh báo:**
   - `suspected`: score đủ cao trong một frame.
   - `confirmed`: tồn tại ổn định qua nhiều frame hoặc có bằng chứng bổ sung.
5. **Không bắt buộc chờ nhiều frame với mọi vật cản:** vật thể gần xe có thể cần cảnh báo sớm. Thời gian xác nhận phải dựa trên khoảng cách, tốc độ và yêu cầu vận hành.
6. **Ghi log:** timestamp, bounding box/mask, score, vị trí trong ảnh, trạng thái phương tiện và ảnh overlay để kiểm tra false positive/false negative.

## 5. Triển khai inference trên server GPU

### Giai đoạn A — Xác nhận khả năng chạy

1. Kiểm tra OS, GPU, driver, CUDA, VRAM và phiên bản Python.
2. Clone repo chính thức của Perspective-Aware Road Obstacle Detection.
3. Làm theo đúng README của repo; cài dependencies trong môi trường riêng.
4. Xác minh checkpoint pretrained có sẵn, điều khoản sử dụng và đường dẫn tải.
5. Chạy inference trên một ảnh trước; xác minh định dạng score, thứ tự màu ảnh, kích thước input/output và bước resize.
6. Chạy model segmentation riêng để xác minh road mask và vehicle mask.

### Giai đoạn B — Tích hợp

1. Viết adapter riêng cho mỗi model, trả về các đầu ra chuẩn:
   - `obstacle_score_map: H x W float`
   - `road_mask: H x W bool`
   - `vehicle_mask: H x W bool`
2. Hợp nhất mask và tạo overlay.
3. Lưu kết quả trên một đoạn video ngắn trước khi xử lý luồng dài.
4. Đo latency từng nhánh, latency tổng, FPS và VRAM.
5. Chỉ tối ưu bằng FP16, batching hoặc TensorRT khi các model và môi trường hỗ trợ, đồng thời kiểm tra lại độ chính xác sau tối ưu.

### Giai đoạn C — Tích hợp service

Đầu vào: video/file hoặc luồng frame có timestamp.  
Đầu ra tối thiểu:
- `timestamp`
- `has_suspected_obstacle`
- `regions`: mask hoặc bounding box
- `score`
- `status`: suspected/confirmed
- `debug_overlay` (tùy chọn)

Cần có timeout, xử lý frame lỗi, log lỗi inference, giám sát GPU/VRAM và cơ chế tránh cảnh báo lặp lại cho cùng một vùng.

## 6. Đánh giá

### Benchmark công khai

Ưu tiên kiểm tra **RoadObstacle21** của SegmentMeIfYouCan vì tập này tập trung vào vật cản trên vùng đường, bất kể class vật thể. Benchmark phân biệt:
- RoadObstacle21: vật cản trên đường.
- RoadAnomaly21: vật thể bất thường về ngữ nghĩa ở bất kỳ vị trí nào trong cảnh.

Links:
- Benchmark: https://segmentmeifyoucan.com/
- Repo đánh giá: https://github.com/SegmentMeIfYouCan/road-anomaly-benchmark

### Tập kiểm thử riêng

Tạo một tập video đại diện cho môi trường thực tế, bao gồm:
- cát, gạch, sỏi, đất, vật liệu xây dựng;
- thùng, túi, hộp, đồ vật rơi;
- xe máy/ô tô đang chạy và đỗ;
- bóng đổ, vạch kẻ, vết vá, ổ gà và mặt đường khác màu;
- đêm, mưa, ngược sáng, rung camera và đường cong.

### Chỉ số cần báo cáo

- **Obstacle recall:** tỷ lệ vật cản thực tế được phát hiện.
- **False positives per minute:** số cảnh báo sai mỗi phút video.
- **Pixel/component metrics:** IoU hoặc các chỉ số đánh giá vùng/đối tượng phù hợp với benchmark.
- **FPS và latency:** đo trên server mục tiêu.
- Phân tích riêng theo kích thước vật cản, khoảng cách tương đối, thời tiết và tình huống xe đỗ.

Không thể suy ra hiệu năng trên video dashcam thực tế chỉ từ kết quả benchmark. Cần kiểm thử theo dữ liệu và tiêu chí vận hành của hệ thống.

## 7. Các giới hạn cần chấp nhận

- Không có model pretrained nào bảo đảm phát hiện mọi vật thể lạ.
- Vật thể có màu sắc/texture gần giống mặt đường có thể bị bỏ sót.
- Vết vá, bóng đổ, ổ gà hoặc thay đổi mặt đường có thể gây cảnh báo sai.
- Sai số của road mask và vehicle mask ảnh hưởng trực tiếp đến kết quả hợp nhất.
- Việc loại trừ xe đồng nghĩa xe đỗ cũng bị loại khỏi cảnh báo, kể cả khi xe đó đang lấn chiếm làn đường.
- Nếu checkpoint Perspective-Aware không chạy được trên môi trường mục tiêu, hãy dùng các phương pháp inference có sẵn trong benchmark làm baseline thay thế trước khi cân nhắc fine-tune.

## 8. Tài liệu tham khảo

1. Perspective-Aware Road Obstacle Detection: https://github.com/cvlab-epfl/perspective-aware-obstacles
2. SegmentMeIfYouCan benchmark: https://segmentmeifyoucan.com/
3. Road anomaly benchmark code: https://github.com/SegmentMeIfYouCan/road-anomaly-benchmark
4. SegFormer documentation: https://huggingface.co/docs/transformers/model_doc/segformer
5. NVIDIA TAO SegFormer Cityscapes: https://catalog.ngc.nvidia.com/orgs/nvidia/tao/models/pretrained_segformer_cityscapes
