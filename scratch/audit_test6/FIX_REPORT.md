# Bản sửa biên đường và vùng mặt đường nghi vấn

Mã chính đã sửa: `run_server_mask2former.py`. Thuật toán hình học và kiểm tra
bề mặt được tách trong `road_geometry.py`. Bản sao trước sửa nằm ở
`run_server_mask2former.before_fix.py` trong thư mục báo cáo này;
`server_fix.diff` ghi thay đổi mã chính.

## Thay đổi

- Bỏ neo hình thang theo tỷ lệ ảnh trong standalone. Lấy thành phần road lớn
  nhất và đo mép trái/phải theo từng hàng. Các hàng có road nhìn thấy không
  bị EMA kéo về vị trí của frame trước.
- Chỉ nối qua đoạn che khuất có điểm đo hỗ trợ. Nếu thiếu một phía, chỉ dùng
  biên frame trước đã warp khi biên đó gần và nằm phía ngoài mép quan sát;
  không kéo đường thẳng xuyên ảnh khi không đủ bằng chứng. Không có road thì
  không tạo hành lang trắng.
- Bù chuyển động cho đường cong của frame trước trước khi dựng biên frame
  hiện tại; dùng cùng số hàng kể cả hàng cuối.
- Tách một phần pixel mang nhãn road nhưng thiếu hỗ trợ semantic, nối với một
  vùng non-road có điểm semantic thấp, đồng thời khớp màu sắc của vùng đó.
  Các pixel này được đánh dấu `surface_uncertain`, không được tô xanh như road
  đã chắc chắn. Pixel road có điểm semantic cao không bị bước này sửa nhãn.
- Tùy chọn `--auto_ego_mask`: loại vùng tiền cảnh lớn nối đáy ảnh, hiệu chỉnh
  bằng tỷ lệ xuất hiện trong
  30 frame đầu. Vùng nền cố định chỉ được mở rộng vào tiền cảnh hiện tại ở
  lân cận vùng camera đã học, không lan sang đống vật liệu phía đối diện.
  Không xác nhận sự kiện trong lúc ROI đang hiệu chỉnh. Có tùy chọn polygon
  riêng theo từng camera.
  **Mặc định tắt** vì vùng đáy ảnh có thể chứa vật thật; không đủ bằng chứng
  để tự coi một vùng cố định là xe camera. Dùng polygon đo theo camera khi có.
- Không miễn trừ sidewalk/terrain chỉ bằng tên nhãn khi xét vật trong hành
  lang. Đồng bộ bicycle/train với danh sách đối tượng được miễn trừ.
- Kiểm tra tiếp xúc đường ở cấp thành phần thay vì cắt mỗi pixel dị vật ở
  khoảng cách 30 pixel. Không ghép contour rời bằng cách nối chuỗi đỉnh;
  không kéo polygon từ contour về tận lề. Loại xe khỏi mask dị vật hiển thị.
- Trong chế độ `--strict_obstacle_support`, biên ngoại suy tự nó không đủ để
  tạo cảnh báo. Ứng viên phải có hỗ trợ trong
  vùng đường đo được, kể cả các lỗ nằm bên trong thành phần road; không chỉ
  nằm trong dải mở rộng do biên ước lượng tạo ra. Điều này có thể bỏ sót vật
  sát lề khi biên hoàn toàn bị che và không có bằng chứng bề mặt.
- Tùy chọn `--strict_obstacle_support`: dải pixel pha trộn sát mép road không
  đủ để xác nhận vật. Cần phần lõi bề
  mặt nghi vấn cách mép semantic road đủ xa, hoặc vùng non-road thực sự nằm
  trong một lỗ được road bao quanh. Đây là điều kiện giảm cảnh báo giả ở mép.
  **Cả hai điều kiện bảo thủ trên mặc định tắt**: kiểm tra A/B của nhóm điều
  kiện này làm giảm số lần theo dõi cát ở test2 mà chưa cải thiện các sự kiện
  test6, nên chưa đủ cơ sở bật mặc định.
- Không vẽ nét trắng trang trí vào mask dữ liệu sau khi đã đục đen dị vật.

## Cách xem kết quả

Chạy:

```bash
python run_server_mask2former.py --input video/test6.mp4 \
  --output_dir output_server_m2f_repaired_test6 --model m2f-large \
  --show_boundary_lines
```

Các tệp xuất:

| Tệp | Ý nghĩa |
|---|---|
| `overlay.mp4` | Xanh: road đã hiệu chỉnh; vàng: mép road đang thấy; cyan: hành lang suy ra; đỏ: vùng dị vật được tracker xác nhận |
| `semantic_road_mask.mp4` | Road từ semantic model, trước bước hiệu chỉnh bề mặt và loại vùng xe quay phim |
| `road_mask.mp4` | Road đã loại vùng không chắc chắn và ROI xe camera |
| `surface_uncertain.mp4` | Pixel nhãn road được đánh dấu nghi vấn bởi bước kiểm tra bề mặt |
| `corridor_mask.mp4` | Hành lang đường kỳ vọng có hỗ trợ quan sát |
| `encroach_mask.mp4` | Hành lang trừ dị vật đã xác nhận, không vẽ đè biên |

Vàng và cyan có thể khác nhau tại vùng nghi bị che hoặc vùng bề mặt nghi vấn:
đó là hai đại lượng khác nhau, không phải hai bản vẽ của cùng mask.
`corridor_mask` không phải ground truth biên thực phía sau vật che.

`--no-surface_refine` tắt bước kiểm tra bề mặt để so sánh. `--auto_ego_mask`
bật ROI tiền cảnh tự động thử nghiệm (mặc định tắt). Để đặt ROI cố định cho
camera, dùng:

```text
--ego_polygon 'x1,y1;x2,y2;x3,y3;...'
```

Tọa độ chuẩn hóa trong [0,1], ít nhất 3 đỉnh; các điểm phải theo thứ tự quanh
polygon. ROI thủ công thay ROI tự động. Cần xác định polygon theo camera thực,
không dùng một polygon chung cho mọi camera.

## Kiểm chứng và giới hạn

Chín kiểm tra trong `test_road_geometry.py` đã qua: giữ mép đường cong không đối
xứng, nội suy đoạn che khuất có điểm neo, không tạo hành lang khi không có
road, không ngoại suy cả ảnh khi thiếu neo, giữ road có hỗ trợ semantic cao,
và chỉ sửa road yếu nối với seed vật liệu phù hợp, có loại vùng người/xe.
ROI camera cũng được kiểm tra để không mở rộng sang vật liệu phía đối diện.
Có kiểm tra seed camera nhô cao trên một dải dashboard ngang và kiểm tra
ROI dùng cache giống hệt kết quả tính trực tiếp.
Có kiểm tra loại dải pha trộn mép không có lõi, đồng thời giữ vật có lõi bề
mặt hoặc nằm trong một lỗ của mask đường.

Vẫn giữ ngưỡng xét dị vật ở nửa dưới ảnh để giảm nhầm hạ tầng/vật xa. Vật
nhỏ hoặc gạch ở xa trên ngưỡng này có thể chưa được xét; không coi là đã phát
hiện hết vật liệu trên toàn bộ khung hình.

Đã chạy cả video test2 và test6 bằng `m2f-large`, xuất kết quả vào các thư mục
mới. Ảnh so sánh và số liệu được lưu cùng báo cáo này.

Các số đo từ frame 974 (không phải ground truth toàn bộ đống cát):

- Ô ảnh `(30,470)–(180,600)` nằm trong vùng cát: road giảm từ 100% còn khoảng
  14,7%. Nghĩa là vẫn còn pixel cát được tô xanh, chưa tách hết vật liệu.
- Dải `y=360..599`: số pixel semantic road nằm ngoài hành lang giảm từ
  75.500 xuống 0. Đây là phép đo mức khớp giữa mask, không chứng minh biên suy
  ra là biên thật phía sau vật che.
- Semantic road gốc giữ nguyên ở các frame đối chiếu; cải thiện đến từ hậu
  xử lý, không phải model mới đã học thêm lớp cát/gạch.

Kiểm tra ROI trên test2: cấu hình tự suy ROI camera chỉ giữ 19 lượt theo dõi
cát; cấu hình mặc định cuối tắt ROI này giữ **94 lượt**, gần 96 lượt ở bản
chẩn đoán ban đầu. Vì vậy ROI camera tự động không được bật mặc định. Sự kiện
test2 bản cuối có peak frame 186, thời gian được quan sát cộng dồn 3,76 giây.

Xem [ảnh test6 frame 974](comparison_test6_974.jpg),
[ảnh test2 frame 190](comparison_test2_190.jpg) và
[số liệu các frame](fix_validation.json).

Ở bản mặc định cuối, test2 cũng khớp tốt hơn: frame 100 giảm số road pixel
ngoài hành lang từ 3.748 xuống 93; frame 190 từ 1.842 xuống 5, cùng dải
`y=360..599`. Thành phần road nhỏ bị tách rời bởi phương tiện không được
coi là thành phần đường chính. Không coi đây là số đo so với ground truth.

Kiểm chứng cuối: test2 đủ 243 frame, khoảng 4,92 FPS, 1 sự kiện; test6 đủ
1.173 frame, khoảng 5,71 FPS, 5 sự kiện. Các sự kiện test6 vẫn có vùng camera
và mép đường cần hậu kiểm; **không kết luận 5 sự kiện là 5 vật lấn chiếm thật**.
Toàn bộ video kết quả đã kiểm tra cùng kích thước 1280×720, tốc độ phát 25 FPS,
đủ số frame. Chi tiết trong `final_video_checks.json`; hash mã nguồn trong
`fix_source_hashes.json`.

Đây là sửa hình học và một bước hậu xử lý vùng không chắc chắn, **không phải
model nhận dạng cát/gạch mới**. Vật liệu bị model gọi là road với điểm cao vẫn
có thể bị bỏ sót. Vật nằm ngoài hành lang quan sát được chưa tự động được kết
luận là lấn chiếm. Khi mép đường bị che hoàn toàn và không có lịch sử đáng tin,
không thể khẳng định vị trí biên thật bằng riêng mask semantic hiện tại.

ROI tự động có thể nhầm vật cố định chạm đáy ảnh ngay từ khi mở video là một
phần xe camera; dùng polygon đã hiệu chỉnh cho triển khai thực. Bước bề mặt
cũng có thể nhầm chỗ đường đổi vật liệu hoặc ánh sáng; kiểm tra false positive
trên camera thực trước khi dùng cảnh báo tự động. Hai video kiểm chứng chưa
đủ để công bố độ chính xác tổng quát hoặc đảm bảo phát hiện hết mọi dị vật.

Các lần kiểm chứng vẫn còn cảnh báo ở mép đường/vùng camera. Số sự kiện
trong JSON không được xem là số vật thật và không dùng để tuyên bố đã hết
nhiễu. Bản sửa này chưa giải quyết triệt để phát hiện vật lạ hoặc khôi phục
biên khi bị che hoàn toàn; cần dữ liệu gán nhãn và tín hiệu nhận dạng/bề mặt
phù hợp để xử lý cả các trường hợp model gán road với độ tin cậy cao.
