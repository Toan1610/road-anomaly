# Kiểm tra pipeline Mask2Former và biên lấn chiếm

> Đây là báo cáo chẩn đoán **trước khi sửa mã**. Xem
> [bản sửa và cách kiểm chứng](FIX_REPORT.md) cho trạng thái hiện tại.

Kiểm tra ngày 02/11/2026 trên `run_server_mask2former.py`, checkpoint
`facebook/mask2former-swin-large-cityscapes-semantic`. Không sửa mã chạy chính,
không ghi đè `output_server_m2f_large_6`.

**Kết luận: pipeline hiện tại có phép lấy phần dư trong hành lang đường, nhưng
chưa giải quyết đầy đủ bài toán phát hiện vật thể lạ lấn chiếm. Biên dựng lại
không bám đúng biên model ở nhiều đoạn không bị che. Nội suy có tồn tại, nhưng
đang phụ thuộc mạnh vào đường tham chiếu mặc định và nhãn semantic.**

## Bằng chứng và cách tái lập

- [Video gốc / overlay cũ / mask cũ cùng frame](contact_sheet.jpg).
- [Test6 frame 950: các bước xử lý](test6/frame_0950.jpg).
- [Test6 frame 974: các bước xử lý](test6/frame_0974.jpg).
- [Test2 frame 190: các bước xử lý](test2/frame_0190.jpg).
- [Số liệu test6](test6/stats.json), [số liệu test2](test2/stats.json).
- [Script chẩn đoán](trace_pipeline.py), chạy từ thư mục gốc:

```bash
venv/bin/python scratch/audit_test6/trace_pipeline.py test6
venv/bin/python scratch/audit_test6/trace_pipeline.py test2
```

Script chạy trực tiếp `main()` gốc với model đã cache, thu mask trung gian bằng
Python trace; tắt ghi video và ảnh bằng chứng để giảm I/O. Vẫn chạy segmentation,
EMA, optical flow, dựng biên, lọc ứng viên và tracking. Ảnh chẩn đoán gồm sáu ô:
ảnh gốc, semantic, hai loại biên, hành lang dựng lại, normal mask, ứng viên trước
khi lọc/tracking. Viền vàng là hành lang; viền tím là contour semantic road.
Các tệp NPZ lưu ảnh, nhãn, điểm semantic, hành lang và ứng viên để đo lại.

Ảnh semantic dùng màu sinh từ bảng tra, không phải bảng màu Cityscapes chuẩn;
chỉ road=xanh lá, sidewalk=cam, terrain=tím được đặt cố định. Không suy nhãn các
màu còn lại bằng mắt; xem trực tiếp `pred` trong NPZ.

SHA256 mã chính lúc chẩn đoán:
`2cdeb70664c7d4f7f887cfe1fce32ae395297734b159ccd987fa0d4fd66199f4`.
Thời gian sửa file nguồn hiện tại muộn hơn các kết quả cũ. Vì vậy không coi ảnh
output cũ và mã hiện tại là cùng một bản chạy. Mã hiện tại có thêm điều kiện loại
nắp xe trong `is_corner_artifact()`; các mẫu nắp xe 5/150/600/900/1123 bị loại khi
chạy lại, dù ảnh output cũ vẫn báo chúng là dị vật.

Tổng được lấy trực tiếp từ `incidents_log` khi `main()` kết thúc: **test6 bản
mã hiện tại có 2 sự kiện**, không phải 0. Một sự kiện bên phải có peak frame
669 (trùng frame bằng chứng báo nhầm phần xe trong output cũ); một sự kiện
bên trái có peak frame 974, chỉ là mảnh cát nói ở dưới. Bộ lọc nắp xe mới làm
giảm báo giả nhưng chưa loại hết. Dòng `Không có sự kiện mới` ở cuối LOOP
chỉ nói về phần flush cuối vòng, không nói tổng cả lần chạy.

Đối chiếu test2 chạy đủ **243/243 frame**, cùng model/tham số, ghi nhận 1 sự
kiện bên trái (peak frame 183, 96 lượt quan sát). Test6 chạy đủ 1.173/1.173
frame. Kiểm tra hash sau khi chạy xác nhận mã nguồn chính không thay đổi.

## Luồng thực tế

1. Co ảnh về 384×384 trên GPU, chạy Mask2Former; gộp các query thành 19 kênh
   semantic rồi nội suy về kích thước video (`infer_mask2former_probs`, dòng 396).
2. EMA điểm semantic, làm mịn kênh road, chọn nhãn lớn nhất cho mỗi pixel.
3. Tạo road, sidewalk, infrastructure và vehicle mask (dòng 2611 trở đi).
4. Dựng hành lang từ hai đường cong (`extract_curved_road_corridor`, dòng 1745).
5. Tạo normal mask: road + sidewalk + người/xe + biển/đèn giao thông. Với
   Mask2Former, pixel không phải road có điểm dưới 0.25 bị gỡ khỏi normal mask
   (dòng 2761). Đóng hình thái bán kính 7 làm mất thêm vùng nhỏ.
6. Ứng viên = trong hành lang AND ngoài normal mask AND y≥50% chiều cao AND
   trong vùng giãn road 30 pixel AND ngoài vehicle mask (dòng 2793).
7. Đóng/mở hình thái, lọc diện tích/góc ảnh/xe/hạ tầng/terrain/vị trí/độ sâu;
   ghép các mảnh rồi tracking tối thiểu khoảng 0.5 giây.
8. Khi track được xác nhận, dựng polygon đặc từ mép hành lang tới contour
   ứng viên. Video mask = hành lang trắng trừ polygon này, rồi vẽ đè lại hai
   đường biên trắng (dòng 3203).

## 1. Biên mask đầu ra chưa bám biên model

Trong standalone, `yolo_lines=None` (dòng 2645). Không có lời gọi fit đường
tham chiếu từ semantic road cho chế độ này. Hàm dựng hành lang tự đặt:

- Đỉnh vùng xử lý: trái = 25% chiều rộng, phải = 75%.
- Đáy vùng xử lý: trái = 8%, phải = 92%.
- Neo cứng 25 hàng đầu vào hai đường này, coi co vào quá 45 pixel là che khuất,
  giới hạn một số đoạn mở rộng, rồi dùng các đường này để khôi phục biên.

Đây là giả định hình học theo ảnh, không phải biên đo được của cảnh. Ví dụ frame
300: trong dải y=360..599 có **81.602 pixel model gán road nằm ngoài hành lang**.
Frame 974 có **75.500 pixel road ngoài hành lang**, đồng thời hành lang chứa
12.360 pixel không phải road trong cùng dải. Đây là số đo chênh lệch giữa hai
mask, không phải sai số so với ground truth. Các ảnh đối chiếu cho thấy biên
vàng cắt vào vùng đường còn nhìn rõ và khác xa contour tím ở nhiều đoạn.

Trên đoạn không bị che, hành lang cần bám mép đường đáng tin cậy. Trên đoạn
bị che, biên kỳ vọng cần đi phía sau vật che, vì vậy không nên yêu cầu nó trùng
contour road đang nhìn thấy ở mọi nơi. Cần tách rõ hai khái niệm này để đánh giá.

## 2. Có nội suy, nhưng chưa đủ tin cậy

Mã đã có phát hiện che khuất, bỏ điểm nghi bị che, `np.interp`, làm mịn Gaussian
và EMA (dòng 1838–1928). Tuy nhiên:

- Việc xác định che khuất và neo biên dựa vào đường mặc định kể trên.
- `mask_infra` gồm sidewalk/fence/vegetation/terrain. Nếu cát bị gán terrain
  hoặc sidewalk, nó có thể bị dùng như bằng chứng mép đường thật; cầu nối bị
  chặn khi tỷ lệ infrastructure trong khoảng trống ≥0.35.
- Nội suy thiếu điểm ở đầu/cuối có thể giữ hằng điểm cuối; đó không phải suy
  luận chắc chắn về hình học đoạn bị che.
- Optical flow hiện warp đường vừa dựng ở frame hiện tại bằng H từ frame trước
  sang hiện tại; đúng ra phải warp đường của frame trước trước khi kết hợp.
  Hơn nữa đường cong có `y_bottom-y_top+1` mẫu, còn nhánh cập nhật dùng
  `np.arange(y_top,y_bottom)` có ít hơn một mẫu. Điều kiện kiểm tra độ dài
  không đạt, nên phần blend hỗ trợ chuyển động không được cập nhật như chú thích.

## 3. Nguyên nhân bỏ sót cát và gạch

Đo trên những ô nhỏ chọn bằng mắt trong ảnh, không phải mask gán nhãn đầy đủ:

| Video / frame | Ô ảnh (x1,y1,x2,y2) | Nhãn semantic | Trong hành lang | Ứng viên sau hình thái |
|---|---|---|---:|---:|
| test6 / 950 | (40,240,180,330), cát | person 89,33%; road 10,59%; car 0,09% | 0% | 0% |
| test6 / 974 | (40,240,180,330), cát | wall 93,33%; road 6,67% | 0% | 0% |
| test6 / 974 | (30,470,180,600), phần cát gần | road 100% | 24,25% | 0% |
| test6 / 974 | (320,80,440,150), vùng gạch | building 77,17%; còn lại road/wall/vegetation/terrain | 0% | 0% |

Hai ô cát phía trên còn nằm trên ngưỡng y=360, nên cả cổng hành lang lẫn cổng
chiều cao đều chặn chúng. Điểm semantic trong ô cát đầu của frame 950 đều dưới
0.25: **không thể nói ô này bị loại chỉ vì gán person**, bởi ngoại lệ điểm thấp
đã gỡ nó khỏi normal mask. Nguyên nhân trực tiếp đo được là ngoài hành lang và
trên ngưỡng chiều cao. Phần cát gần ở frame 974 bị gán road thì không được gỡ
khỏi normal mask, dù điểm thấp, nên bị loại ngay từ phép lấy phần dư.

Vùng gạch trong mẫu ở xa, cao hơn ngưỡng xử lý và ngoài hành lang. Không có
ground truth biên/footprint để kết luận toàn bộ đống gạch thực sự nằm trong
lòng đường hay chỉ ở lề. Các số liệu này chứng minh pipeline chưa xét vùng đó,
không chứng minh mọi pixel của đống gạch đều là lấn chiếm.

Frame 974 vẫn có một ứng viên rời: bbox (197,361,201,97), diện tích contour
11.544 px², độ sâu theo biên đang dựng 200,4 px. Nó không đại diện cho toàn
bộ đống cát. Ở các mẫu 900/930/950/1000/1030 không có ứng viên được giữ lại.

Trong test2, frame 150/170/190 có ứng viên khoảng 6.157 / 14.267 / 30.305 px².
Đống cát lớn cắt qua hành lang giả định và vùng gần road mask, nên có phần dư
liên tục hơn. Điều này giải thích sự khác nhau giữa hai video; không chứng minh
test2 đã khôi phục đúng toàn bộ dị vật. Frame 190 còn có thành phần ứng viên
kéo dài xuống đáy ảnh.

Những điểm làm mất thêm ứng viên:

- Sidewalk được coi là normal kể cả khi nằm trong hành lang.
- Loại contour khi nhãn infrastructure chiếm >40%; bên trái còn loại khi
  terrain chiếm >40% (dòng 2867–2876). Đây là lọc bằng nhãn, không phải kiểm
  chứng vật đó nằm ngoài đường.
- Điều kiện từng pixel phải cách road không quá khoảng 30 pixel chỉ lấy lớp
  sát bề mặt đường nhìn thấy, không giữ toàn bộ đống vật liệu lớn.
- Vehicle LUT thiếu bicycle (18), trong khi normal mask có bicycle. Ngược lại
  train (16) có trong LUT phương tiện nhưng không có trong CORRIDOR_CLASSES.
- Việc ghép contour bằng `np.vstack` nối hai chuỗi điểm độc lập; khi vẽ đặc có
  thể tạo cạnh nối giả. `get_solid_obstacle_mask` lại kéo từ contour về lề,
  nên mask hiển thị có thể rộng hơn vùng bằng chứng thực tế.

## 4. Vì sao có nhiễu và vì sao log là 1 nhưng tổng là 4?

Ảnh output cũ các frame 150/750/900/1123 báo đỏ phần xe quay phim ở góc dưới
phải. Đây là báo động giả. Cắt ở 90%/91% chiều cao không đủ, vì phần xe vươn
lên cao hơn các ngưỡng này. Mã hiện tại đã thêm bộ lọc bbox loại các mẫu đó;
lọc theo bbox vẫn có nguy cơ loại cả vật thật ở vị trí tương tự.

`static_only` thực tế chủ yếu loại vùng nhãn người/xe và vùng đệm, không kiểm
tra vận tốc vật sau bù chuyển động camera. EMA và tracking trong tọa độ ảnh
chưa đủ bảo đảm vật thể tĩnh hoặc hết rung/nhiễu.

`encroach_mask.mp4` không phải mask nhị phân thô của model: chỉ đục đen dị vật
đã xác nhận, giãn polygon về lề, rồi vẽ đè viền trắng. Vì vậy vùng trắng sạch
không đồng nghĩa model đã xác định đúng đường; nét trắng tách rời quanh vùng
đen có thể đến từ bước vẽ biên. Nén MP4 và anti-alias cũng không giữ giá trị
pixel tuyệt đối 0/255; khi đo phải threshold hoặc dùng NPZ/PNG.

`LOOP ... 1 sự kiện` đếm track được flush ở cuối vòng, còn tổng 4 đếm cả các
sự kiện đã ghi trước đó. Hai dòng không cùng phạm vi đếm (dòng 2409–2461 và
3290–3304). Không dùng số ở dòng LOOP làm tổng sự kiện cả video.

## 5. Cách diễn đạt đúng bài toán và thứ tự sửa

**Không thể loại hết mọi class model rồi kỳ vọng phần còn lại là unknown.**
Ở luồng này, `argmax` luôn chọn trong 19 kênh; pixel cát vẫn nhận road/person/
wall/... Mã không tạo bản đồ unknown riêng. Kênh null query bị bỏ lúc gộp
semantic, không phải một lớp unknown ở từng pixel. Điểm tổng các query dùng
ở đây cũng không phải xác suất đã hiệu chuẩn để quyết định OOD bằng một
ngưỡng duy nhất. Đặc biệt cát bị gán road tự tin hoặc thiếu tự tin vẫn bị
coi là normal theo điều kiện hiện tại.

Phép loại trừ phù hợp là:

```text
ứng viên hình học = hành lang đường kỳ vọng
                  trừ (mặt đường thực sự nhìn thấy + đối tượng được miễn trừ)

xác nhận lấn chiếm = bằng chứng vật thể không phải mặt đường
                    + vị trí tiếp xúc/footprint trong hành lang
                    + độ tin cậy biên và tính nhất quán theo thời gian
```

Hạ tầng ngoài hành lang tự được loại bằng vị trí. Hạ tầng/vật liệu nằm trong
hành lang cần được xem xét, không miễn trừ chỉ vì model gọi nó sidewalk,
terrain hay wall. Pixel đã bị gán road sai cần một tín hiệu độc lập về vật thể/
bề mặt hoặc dữ liệu huấn luyện phù hợp; thay ngưỡng hậu xử lý không tạo lại
thông tin mà semantic model không phân biệt được.

Ưu tiên sửa:

1. Dựng hai biên từ các đoạn mép đường nhìn thấy đáng tin cậy của chính cảnh;
   bỏ neo hình thang cố định trong standalone. Có trạng thái observed / inferred /
   uncertain cho từng đoạn; tránh dùng bản thân dị vật làm neo biên.
2. Định nghĩa vùng xe quay phim theo camera và dùng cùng ROI cho detection,
   tính độ tin cậy và hiển thị; không chỉ lọc bbox sau khi ứng viên đã hình thành.
3. Tách mask road nhìn thấy, hành lang kỳ vọng và mask dị vật. Xuất mask dữ liệu
   không vẽ nét trang trí; vẽ biên quan sát/nội suy riêng trên overlay chẩn đoán.
4. Chỉ loại nhất quán các nhóm người/xe được yêu cầu miễn trừ. Không loại cát
   chỉ vì nó mang nhãn terrain/sidewalk. Chuyển điều kiện road-contact thành
   kiểm tra kết nối/tiếp xúc của thành phần, thay vì cắt từng pixel ở bán kính 30.
5. Thêm bằng chứng phân biệt cát/gạch với road và hợp nhất mask bằng phép OR
   hình học thay vì nối chuỗi contour; sửa warp/EMA trước khi làm tracking tĩnh.
6. Gán nhãn kiểm định trên cả test2 và test6: biên rõ, biên bị che, footprint cát/
   gạch, vùng xe camera; đo lỗi biên, recall dị vật và báo động giả. Ảnh chẩn
   đoán hiện tại chưa phải bộ ground truth và chưa đủ để tuyên bố độ chính xác.

Không khuyến nghị chỉ giảm `min_encroach_area`, `min_encroach_depth` hoặc
`min_duration`: những vùng đã mất ở semantic/ROI/hành lang không quay lại,
trong khi số báo động giả có thể tăng.
