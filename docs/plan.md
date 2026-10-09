# Kế hoạch tổng thể — Giai đoạn 0

_Phiên bản 1 · 09/10/2026 · Trạng thái: đã duyệt_

Đây là kế hoạch tham chiếu của đồ án. Mọi thay đổi về phạm vi, phương pháp hoặc mốc nghiệm thu phải được ghi lại kèm lý do trong `PROJECT_STATUS.md`.

## 1. Đề bài

### 1.1 Bài toán

Đầu vào là một tập ảnh hoặc một video quay không gian nội thất tĩnh bằng camera thông thường, không dùng LiDAR hay cảm biến độ sâu. Hệ thống cần:

1. Ước lượng tham số camera và cấu trúc thưa.
2. Tối ưu biểu diễn 3D Gaussian Splatting cho toàn bộ không gian.
3. Cho phép quan sát và di chuyển trong cảnh ở tốc độ tương tác.
4. Đánh giá chất lượng bằng một giao thức thực nghiệm lặp lại được.

### 1.2 Mục tiêu cụ thể

| Mã | Mục tiêu |
|---|---|
| O1 | Pipeline ảnh/video → chọn frame → camera và điểm thưa → 3DGS → xuất cho viewer, chạy bằng một lệnh kèm file cấu hình |
| O2 | Tái lập 3DGS trên cảnh nội thất công khai, sai lệch so với số liệu công bố nằm trong ngưỡng chấp nhận |
| O3 | 3–5 không gian tự quay, có độ khó khác nhau, theo quy trình quay chuẩn hóa |
| O4 | 1 cải tiến (tối đa 2) nhắm vào một vấn đề nội thất cụ thể, có giả thuyết, ablation và phân tích thất bại |
| O5 | Viewer 3D tương tác, ưu tiên chạy trên web, có số đo thời gian tải và FPS |
| O6 | Mọi con số trong báo cáo truy được về file kết quả, commit và cấu hình tương ứng |

### 1.3 Đầu vào (mặc định)

| Hạng mục | Mặc định | Ghi chú |
|---|---|---|
| Thiết bị | Camera chính của smartphone hoặc máy ảnh thường | Không dùng LiDAR/depth kể cả khi máy có |
| Dạng dữ liệu | Video MP4/MOV hoặc ảnh JPG/HEIC | Video thuận tiện hơn; ảnh chụp thường nét hơn |
| Quy mô | Một phòng hoặc một không gian liên thông | Nhiều phòng là phần mở rộng |
| Số frame | Vài trăm frame sau chọn lọc | Hiệu chỉnh theo diện tích và GPU |
| Điều kiện quay | Cảnh tĩnh, ánh sáng ổn định, khóa phơi sáng và cân bằng trắng nếu được | Không có người di chuyển |

### 1.4 Đầu ra

| Đầu ra | Mức | Định dạng dự kiến |
|---|---|---|
| Camera và đám mây điểm thưa | Bắt buộc (để kiểm tra) | Định dạng COLMAP |
| Mô hình 3DGS | Bắt buộc | `.ply` chuẩn, kèm bản nén cho web |
| Viewer tương tác | Bắt buộc | Trang web chạy trên trình duyệt desktop |
| Báo cáo đánh giá | Bắt buộc | Bảng chỉ số, ảnh render so với ảnh thật, log cấu hình |
| Video quỹ đạo render | Nên có | MP4, dùng cho slide và dự phòng khi demo |
| Depth map / mesh | Tùy GVHD | Chỉ làm khi đề tài yêu cầu hình học |
| Kích thước thật (mét) | Ngoài phạm vi mặc định | Cần vật tham chiếu tỉ lệ |

### 1.5 Phạm vi

- **Trong phạm vi:** cảnh tĩnh trong nhà; xử lý offline trên GPU NVIDIA; đánh giá định lượng và định tính; viewer web.
- **Ngoài phạm vi:** cảnh động; tái tạo thời gian thực trên điện thoại; giải pháp phụ thuộc LiDAR; cảnh ngoài trời; phân đoạn hoặc chỉnh sửa ngữ nghĩa; relighting; huấn luyện mô hình feed-forward từ đầu; đo kích thước chính xác khi không có vật tham chiếu.
- **Mở rộng có điều kiện** (chỉ khi các mốc bắt buộc đã đạt): nhiều phòng, trích mesh, upload qua web có hàng đợi xử lý, xem trên mobile hoặc VR.

### 1.6 Lưu ý khái niệm

1. **Render đẹp không đồng nghĩa với hình học đúng.** 3DGS tối ưu theo sai số ảnh. Nếu cần mô hình đo được, phải đổi cả phương pháp lẫn chỉ số (Chamfer, F-score trên ground truth).
2. **Thiếu tỉ lệ tuyệt đối.** SfM từ ảnh camera thường chỉ khôi phục cảnh sai khác một hệ số tỉ lệ; muốn có đơn vị mét cần vật hoặc marker biết trước kích thước.
3. **Chất lượng giảm khi nhìn xa quỹ đạo quay.** Demo cần giới hạn vùng di chuyển hoặc có sẵn các điểm nhìn đặt trước.

## 2. Câu hỏi cần xác minh với GVHD

Mức "Chặn" nghĩa là khi chưa có câu trả lời thì chưa chốt được phần kế hoạch liên quan.

| # | Mức | Câu hỏi | Trạng thái / giả định tạm |
|---|---|---|---|
| Q1 | Chặn | Kết quả chính là hiển thị chân thực hay hình học đo được (mesh, kích thước)? | Tạm: ưu tiên hiển thị (NVS), hình học là phần phụ |
| Q2 | Chặn | Đồ án thiên về hệ thống ứng dụng hay bắt buộc cải tiến thuật toán? "Mới" ở mức nào? | Tạm: hệ thống hoàn chỉnh + 1 cải tiến có kiểm chứng |
| Q3 | Chặn | Các mốc chính thức: nộp đề cương, giữa kỳ, nộp báo cáo, bảo vệ | Tạm: 16 tuần |
| Q4 | Chặn | Tài nguyên GPU | **Đã chốt: Runpod.** Loại GPU chọn theo tiêu chí trong `PROJECT_STATUS.md` |
| Q5 | Quan trọng | Làm cá nhân hay theo nhóm? | Tạm: cá nhân |
| Q6 | Quan trọng | Biểu mẫu, ngôn ngữ, chuẩn trích dẫn, số trang, quy định khai báo dùng AI | Tạm: tiếng Việt, chuẩn IEEE |
| Q7 | Quan trọng | Có bắt buộc dữ liệu tự thu? Được dùng dataset phải đăng ký (ScanNet++)? | Tạm: kết hợp dataset công khai và dữ liệu tự quay |
| Q8 | Quan trọng | Quy định về quyền riêng tư khi quay nhà hoặc văn phòng thật | Tạm: chỉ quay nơi được phép; không công bố dữ liệu thô |
| Q9 | Quan trọng | Demo ở dạng nào? Có bắt buộc luồng upload → tự xử lý → xem? Máy demo có GPU/Internet? | Tạm: CLI xử lý offline + viewer web chạy offline |
| Q10 | Quan trọng | Cần so sánh với bao nhiêu phương pháp khác? Có cần khảo sát người dùng? | Tạm: baseline + 1–2 phương pháp; không khảo sát |
| Q11 | Quan trọng | Được dùng code có license phi thương mại và mô hình pretrained không? | Tạm: được dùng, ghi rõ license từng thành phần |
| Q12 | Bổ sung | Có yêu cầu bài báo, poster hoặc video không? | Tạm: không |

## 3. Phương pháp và quy trình

### 3.1 Luồng xử lý dự kiến

```
Ảnh / video
 → Kiểm tra & chọn frame   (độ nét, trùng lặp, phơi sáng)
 → Ước lượng camera         (COLMAP tăng dần / toàn cục; dự phòng: VGGT)
 → Huấn luyện 3DGS          (baseline → biến thể cải tiến)
 → Đánh giá                 (PSNR/SSIM/LPIPS, thời gian, VRAM, dung lượng)
 → Xuất & nén               (.ply → định dạng web)
 → Viewer web
```

### 3.2 Giai đoạn và cổng quyết định

| Giai đoạn | Hoạt động chính | Cổng để đi tiếp |
|---|---|---|
| 0 | Kế hoạch này; trả lời Q1–Q12 | GVHD đồng ý phạm vi, đầu ra, lịch và tài nguyên |
| 1 | Literature matrix; đọc paper gốc; so sánh theo tiêu chí ở mục 3.3 | Có văn bản quyết định bản cài đặt baseline và tối đa 2 hướng cải tiến |
| 2 | Kiểm tra GPU thật; khóa phiên bản; tải dataset; quy trình quay; quay thử | Dựng lại được môi trường từ đầu; SfM chạy được trên cảnh công khai và cảnh quay thử |
| 3 | Pipeline tối thiểu end-to-end, có cả viewer thô | Baseline đạt ngưỡng; mô hình tự quay xem được trên web |
| 4 | Phân tích thất bại → giả thuyết → cài đặt → ablation | Có so sánh cùng giao thức, kể cả khi không cải thiện |
| 5 | CLI hoàn chỉnh, nén mô hình, viewer, (tùy chọn) giao diện upload | Chạy được end-to-end trên video mới theo README |
| 6 | Tổng hợp, viết báo cáo, kiểm tra nhất quán, chuẩn bị demo | Mọi số liệu truy được; demo đã diễn tập trên máy bảo vệ |

Viewer được đưa vào ở dạng thô ngay từ giai đoạn 3 để phát hiện sớm lỗi định dạng, hệ tọa độ và dung lượng.

### 3.3 Baseline

- **Phương pháp (đã chốt):** 3DGS gốc (Kerbl et al., 2023).
- **Tiêu chí chọn bản cài đặt:** phù hợp nội thất; code chính chủ và license cho phép; chạy được trên GPU thực tế; tương thích phiên bản đã khóa; tái lập được số liệu công bố; xuất được sang viewer; dễ mở rộng.
- **B0 – bản chính chủ Inria:** mốc đối chiếu số liệu. Inria khuyến nghị 24 GB VRAM để huấn luyện đạt chất lượng như paper; môi trường mặc định dựa trên CUDA 11; license chỉ cho phép dùng phi thương mại cho nghiên cứu/đánh giá.
- **B1 – 3DGS trên gsplat (Apache-2.0):** nền phát triển dễ mở rộng. Chỉ dùng thay B0 khi đã tự chứng minh số liệu tương đương trên cùng cảnh. B0 và B1 có thể cần hai môi trường riêng vì khác phiên bản PyTorch/CUDA.

### 3.4 Hướng cải tiến ứng viên

Chọn sau khi phân tích thất bại của baseline ở giai đoạn 4.

| Vấn đề nội thất | Giả thuyết | Kỹ thuật tham khảo | Cách đo |
|---|---|---|---|
| Tường, trần trơn gây floater và hình học sai | Ràng buộc độ sâu/pháp tuyến từ mô hình đơn ảnh giảm floater mà không làm giảm PSNR | DN-Splatter, GaussianRoom, 2DGS-Room | Sai số độ sâu so với ground truth, PSNR/LPIPS, định tính |
| Phơi sáng và cân bằng trắng tự động | Mô hình hóa ngoại hình theo từng ảnh giảm vệt màu | Bù affine (Inria), bilateral grid / PPISP (gsplat) | PSNR, định tính ở vùng sáng–tối |
| Video rung, mờ | Chọn frame theo độ nét và độ phủ tốt hơn lấy mẫu đều | Lọc theo độ nét; Seiskari et al., ECCV 2024 | Tỉ lệ ảnh đăng ký SfM, PSNR, thời gian |
| SfM thất bại ở cảnh khó | Đặc trưng học sâu hoặc mô hình feed-forward cứu được cảnh mà SIFT thất bại | ALIKED + LightGlue (COLMAP 4), VGGT | Tỉ lệ đăng ký, sai số pose, PSNR ở bước sau |

**Ranh giới trung thực:** repo Inria đã có sẵn depth regularization và bù phơi sáng (bản cập nhật 10/2024). Chỉ bật các tùy chọn này thì không phải đóng góp. Đóng góp hợp lệ phải là:

- (a) một thay đổi có giả thuyết rõ ràng, vượt được baseline *đã bật sẵn* các tùy chọn đó; hoặc
- (b) đánh giá có hệ thống các kỹ thuật này trên dữ liệu nội thất quay bằng điện thoại, kèm pipeline và giao thức đánh giá.

## 4. Đánh giá

### 4.1 Bộ chỉ số

| Nhóm | Chỉ số | Dữ liệu cần | Ghi chú |
|---|---|---|---|
| Chất lượng hiển thị | PSNR, SSIM, LPIPS trên ảnh giữ lại | Ảnh test có pose | Chỉ số chính |
| Ước lượng camera | Tỉ lệ ảnh đăng ký, sai số chiếu lại; sai số pose nếu có ground truth | Pose ground truth (ScanNet++) | Kiểm tra trước khi huấn luyện |
| Hình học (nếu Q1 yêu cầu) | AbsRel/RMSE độ sâu, Chamfer, F-score | Laser scan ground truth | Không suy ra từ PSNR |
| Hiệu năng | Thời gian huấn luyện, VRAM đỉnh, số Gaussian, dung lượng file | Log | Luôn ghi loại GPU |
| Sản phẩm | Thời gian end-to-end, số bước thủ công, thời gian tải, FPS | Máy demo | Ghi cấu hình máy |
| Định tính | Ảnh so sánh, video quỹ đạo, thư viện các ca thất bại | — | Luôn đi kèm số liệu |

### 4.2 Giao thức thực nghiệm

- **Điều kiện so sánh:** cùng pose camera, cùng độ phân giải, cùng số vòng lặp, cùng GPU.
- **Dataset công khai:** dùng đúng cách chia train/test của paper (3DGS gốc dùng cách chia kiểu Mip-NeRF 360). Đối chiếu với số liệu của đúng phiên bản code, vì mô hình pretrained của Inria cho chỉ số khác paper do codebase đã được sửa lỗi.
- **Video tự quay dễ rò rỉ dữ liệu:** frame test sát frame train gần như trùng nhau, làm điểm số bị thổi phồng. Quay thêm một đoạn quỹ đạo riêng để kiểm tra, hoặc tách tập test theo các đoạn cách xa nhau.
- **Nguồn số liệu:** luôn tách bạch số liệu công bố trong paper với số liệu tự chạy.
- **Kết luận "cải thiện":** chạy lặp nhiều seed (đề xuất ≥ 3) trên ít nhất một tập con.
- **Lưu vết:** mỗi lần chạy lưu commit hash, file cấu hình, thông tin môi trường, log và chỉ số dạng JSON/CSV.

### 4.3 Trạng thái hoàn thành

`Đã viết` → `Đã chạy` → `Đã kiểm thử` → `Đã xác nhận đạt mục tiêu`. Chỉ mức cuối cùng mới gọi là "hoàn thành".

## 5. Mốc nghiệm thu

Lịch giả định 16 tuần, điều chỉnh theo câu trả lời Q3. Một mốc chưa đạt tiêu chí thì không chuyển sang mốc sau, trừ các việc ghi là chạy song song.

| Mốc | Tuần | Đầu ra | Tiêu chí nghiệm thu |
|---|---|---|---|
| M0 Chốt đề bài | 1 | Bản phạm vi trả lời Q1–Q12; đăng ký ScanNet++ | GVHD xác nhận đầu ra chính, mức đóng góp, lịch, GPU |
| M1 Khảo sát, chọn bản cài đặt | 1–3 | Literature matrix (≥ 25 công trình, ≥ 10 đọc kỹ); BibTeX đã kiểm chứng; văn bản quyết định | Mỗi mục có nguồn chính thức; quyết định so sánh đủ các tiêu chí ở 3.3; GVHD đồng ý |
| M2 Môi trường, dữ liệu | 2–4 (song song M1) | File khóa phiên bản; báo cáo GPU thực tế; ≥ 2 cảnh nội thất công khai; quy trình quay; 1 cảnh quay thử | Dựng lại môi trường thành công từ đầu; SfM chạy được, có tỉ lệ đăng ký |
| M3 Baseline end-to-end | 4–6 | Pipeline từ video đến viewer (cho phép thao tác tay); bảng kết quả baseline | Chỉ số trên cảnh công khai lệch so với số công bố trong ngưỡng (đề xuất ≤ 0,5 dB PSNR, chốt khi có số liệu gốc); cảnh quay thử xem được trên web; đủ log và cấu hình |
| M4 Dữ liệu tự quay, phân tích thất bại | 6–7 | 3–5 cảnh theo quy trình; bảng baseline; thư viện ca thất bại; giả thuyết cải tiến | Mỗi cảnh có chỉ số và ảnh minh họa; giả thuyết được GVHD duyệt |
| M5 Cải tiến, ablation | 7–11 | Cài đặt cải tiến; bảng so sánh và ablation; chi phí thời gian và VRAM | Cùng giao thức với baseline; ≥ 3 cảnh; phân tích cả trường hợp không cải thiện |
| M6 Sản phẩm demo | 9–13 (song song M5) | CLI chạy một lệnh; viewer web với mô hình nén; hướng dẫn sử dụng | Người khác chạy được trên video mới theo README; có số đo end-to-end, tải, FPS; liệt kê bước còn thủ công |
| M7 Đánh giá tổng hợp, báo cáo | 12–15 | Bảng, hình, báo cáo theo biểu mẫu | Mọi số liệu truy được tới file kết quả và commit; mục tiêu, phương pháp, code, kết quả nhất quán |
| M8 Bảo vệ | 15–16 | Slide, kịch bản demo, video dự phòng, câu hỏi dự kiến | Diễn tập trọn vẹn trên máy bảo vệ, có phương án offline |

**Phân tầng ưu tiên**

- **Bắt buộc:** M0–M8; M5 có ít nhất một hướng cải tiến (kết quả âm vẫn được chấp nhận nếu phân tích đầy đủ); M6 tối thiểu gồm CLI và viewer.
- **Nên có:** nén mô hình cho web; thêm một phương pháp so sánh.
- **Chỉ khi dư thời gian:** hướng cải tiến thứ hai, nhiều phòng, trích mesh, giao diện upload, mobile/VR.

## 6. Rủi ro

| ID | Nhóm | Rủi ro | Giảm thiểu / dự phòng |
|---|---|---|---|
| R1 | Dữ liệu | SfM thất bại do tường trơn, texture lặp, ít chồng lấp | Quy trình quay; đặc trưng học sâu, SfM toàn cục; dự phòng VGGT |
| R2 | Dữ liệu | Mờ chuyển động, rolling shutter, phơi sáng thay đổi | Khóa phơi sáng; lọc frame theo độ nét; bù ngoại hình |
| R3 | Dữ liệu | Người di chuyển, gương, kính, màn hình | Quy tắc quay; mask nếu cần; ghi nhận là giới hạn |
| R4 | Dữ liệu | Chậm được cấp quyền ScanNet++ | Đăng ký sớm; dự phòng cảnh trong nhà của Mip-NeRF 360 và Deep Blending |
| R5 | Dữ liệu | Quyền riêng tư khi quay và đưa video lên cloud | Xin phép; dọn vật nhạy cảm; không công bố dữ liệu thô |
| R6 | GPU | Thiếu VRAM | Giảm độ phân giải; giới hạn số Gaussian; dùng B1 |
| R7 | GPU | GPU không sẵn có hoặc chi phí vượt dự kiến | Network volume ở data center có nhiều GPU phù hợp; Stop pod khi không dùng; theo dõi số dư |
| R8 | GPU | Lỗi build CUDA extension | Khóa phiên bản; kiểm tra ngay khi cài môi trường |
| R9 | Thời gian | Lan man thử nhiều biến thể; một người ôm nhiều việc | Giới hạn thời gian giai đoạn 1; tối đa 2 hướng cải tiến; MVP trước |
| R10 | Thời gian | Huấn luyện lâu, thực nghiệm bị dồn | Ngân sách thực nghiệm; thử ở độ phân giải thấp trước |
| R11 | Thời gian | Báo cáo viết muộn, lệch với code | Viết song song; bảng sinh trực tiếp từ file kết quả |
| R12 | Chất lượng | Artifact khi nhìn xa quỹ đạo quay | Quay phủ đủ góc; giới hạn vùng di chuyển; nêu rõ trong báo cáo |
| R13 | Chất lượng | Cải tiến không vượt baseline | Phân tích thất bại vẫn là kết quả hợp lệ; có hướng dự phòng |
| R14 | Chất lượng | Chỉ số và cảm nhận thị giác lệch nhau | Báo cáo đủ 3 chỉ số kèm đánh giá định tính |
| R15 | Tích hợp | File quá nặng cho web | Nén mô hình, giảm bậc SH, giới hạn số Gaussian |
| R16 | Tích hợp | Lệch hệ tọa độ hoặc định dạng giữa trainer và viewer | Kiểm thử chuyển đổi bằng cảnh mẫu ngay giai đoạn 3 |
| R17 | Tích hợp | Demo hỏng khi bảo vệ | Mô hình tính sẵn, viewer offline, video dự phòng |
| R18 | Học thuật | Phóng đại đóng góp | Rà checklist các tuyên bố trước khi viết |
| R19 | Pháp lý | Vi phạm license của code hoặc mô hình pretrained | Bảng license từng thành phần; không chép code nếu không cần |

## 7. Quản lý code

- Chủ repository quyết định thời điểm commit và push; không viết lại lịch sử khi chưa có yêu cầu.
- Commit không chèn dòng đồng tác giả hay thông tin phiên làm việc của công cụ AI.
- Không đưa dataset, checkpoint, `.ply`, video, log thô, token hay file `.env` lên Git.
- Nhánh `main` luôn chạy được; commit message dạng `loại(phạm vi): mô tả`.

## Nguồn đã kiểm tra (09/10/2026)

- graphdeco-inria/gaussian-splatting — README và LICENSE: https://github.com/graphdeco-inria/gaussian-splatting
- nerfstudio-project/gsplat: https://github.com/nerfstudio-project/gsplat
- COLMAP — CHANGELOG (bản 4.0.0 tích hợp GLOMAP, thêm ALIKED và LightGlue): https://github.com/colmap/colmap/blob/main/CHANGELOG.rst
- facebookresearch/vggt: https://github.com/facebookresearch/vggt
- ScanNet++: https://scannetpp.mlsg.cit.tum.de/scannetpp/
- Runpod — Storage: https://docs.runpod.io/pods/storage
- Runpod — Network volumes: https://docs.runpod.io/storage/network-volumes
