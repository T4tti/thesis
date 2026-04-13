# Lý thuyết đề tài (bản rút gọn)

## 1. Bối cảnh và mục tiêu nghiên cứu

Đề tài tập trung vào bài toán dự báo xếp hạng tín nhiệm doanh nghiệp từ dữ liệu tài chính theo thời gian.

Mục tiêu chính:
- Xây dựng pipeline dữ liệu có thể tái lập từ tiền xử lý đến huấn luyện mô hình.
- Chuẩn hóa dữ liệu từ nhiều nguồn về cùng cấu trúc để giảm sai lệch giữa nguồn.
- Giảm tác động của mất cân bằng lớp trong bài toán phân loại tín nhiệm.
- Nâng chất lượng dự báo nhờ khai thác tín hiệu tài chính, tín hiệu thời gian và bối cảnh ngành.

## 2. Ý nghĩa các thuộc tính trong bộ dữ liệu 3 nhóm

Bộ dữ liệu sử dụng file `data/processed/merged_credit_rating_common_3groups.csv`, gồm các nhóm biến sau.

### 2.1 Nhóm định danh và bối cảnh

- `rating_detail`: Nhãn mục tiêu 3 nhóm gồm `IG`, `HY`, `Distressed`.
- `company_name`: Tên doanh nghiệp, chủ yếu dùng để tra cứu.
- `ticker`: Mã doanh nghiệp, quan trọng để gom chuỗi theo thực thể.
- `rating_agency`: Tổ chức xếp hạng, phản ánh khác biệt chuẩn chấm điểm giữa các agency.
- `rating_date`: Thời điểm đánh giá, dùng tạo đặc trưng thời gian và sắp thứ tự chuỗi.
- `sector`: Ngành doanh nghiệp, dùng cho chuẩn hóa theo ngành và phân tích rủi ro theo lĩnh vực.
- `source`: Nguồn dữ liệu sau hợp nhất, dùng để kiểm soát độ lệch giữa nguồn.

### 2.2 Nhóm chỉ số tài chính cốt lõi

- `current_ratio` (Hệ số thanh toán hiện hành): Công thức = Tài sản ngắn hạn / Nợ ngắn hạn. Chỉ số này đo sức khỏe thanh khoản ngắn hạn; thường <1 là tín hiệu áp lực thanh khoản, quá cao kéo dài có thể phản ánh sử dụng vốn lưu động chưa hiệu quả.
- `debt_equity_ratio` (Hệ số nợ trên vốn chủ sở hữu): Công thức = Tổng nợ / Vốn chủ sở hữu. Đây là thước đo đòn bẩy tài chính; giá trị cao làm tăng độ nhạy với biến động lãi suất và chu kỳ kinh doanh, giá trị âm thường đi kèm vốn chủ âm và rủi ro cấu trúc vốn.
- `gross_profit_margin` (Biên lợi nhuận gộp): Công thức = Lợi nhuận gộp / Doanh thu thuần. Dùng để đánh giá năng lực tạo lợi nhuận từ hoạt động sản xuất - bán hàng trước chi phí quản lý và tài chính.
- `operating_profit_margin` (Biên lợi nhuận hoạt động): Công thức = Lợi nhuận hoạt động / Doanh thu thuần. Phản ánh hiệu quả hoạt động cốt lõi; giảm liên tục thường là cảnh báo suy yếu năng lực cạnh tranh hoặc chi phí vận hành tăng.
- `ebit_margin` (Biên lợi nhuận EBIT): Công thức = EBIT / Doanh thu thuần. Là chỉ báo gần với năng lực trả lãi vay; EBIT margin thấp hoặc âm làm tăng xác suất bị hạ bậc tín nhiệm khi chi phí vốn tăng.
- `pretax_profit_margin` (Biên lợi nhuận trước thuế): Công thức = Lợi nhuận trước thuế / Doanh thu thuần. Cho biết mức sinh lời trước tác động thuế, hữu ích để so sánh doanh nghiệp có chính sách thuế khác nhau.
- `net_profit_margin` (Biên lợi nhuận ròng): Công thức = Lợi nhuận sau thuế / Doanh thu thuần. Là kết quả cuối cùng sau mọi chi phí; biên ròng thấp hoặc âm kéo dài thường gắn với rủi ro tín dụng cao hơn.
- `asset_turnover` (Vòng quay tài sản): Công thức = Doanh thu thuần / Tổng tài sản bình quân. Chỉ số hiệu quả sử dụng tài sản; cần đánh giá theo ngành vì mô hình kinh doanh thâm dụng tài sản thường có vòng quay thấp hơn.
- `roe` (Tỷ suất sinh lời trên vốn chủ sở hữu): Công thức = Lợi nhuận sau thuế / Vốn chủ sở hữu bình quân. ROE cao và ổn định thường tích cực, nhưng ROE cao do đòn bẩy quá mức có thể đi kèm rủi ro tài chính cao.
- `roa` (Tỷ suất sinh lời trên tổng tài sản): Công thức = Lợi nhuận sau thuế / Tổng tài sản bình quân. Thể hiện hiệu quả tạo lợi nhuận từ toàn bộ tài sản; ROA âm liên tục là tín hiệu suy yếu chất lượng tài sản và dòng tiền.
- `operating_cashflow_ps` (Dòng tiền hoạt động trên mỗi cổ phần): Công thức = Dòng tiền thuần từ hoạt động kinh doanh / Số cổ phần lưu hành. Chỉ số này phản ánh chất lượng lợi nhuận; dương và ổn định thường tốt cho khả năng trả nợ.
- `free_cashflow_ps` (Dòng tiền tự do trên mỗi cổ phần): Công thức xấp xỉ = (Dòng tiền hoạt động - Chi tiêu vốn) / Số cổ phần lưu hành. Đây là phần tiền còn lại để trả nợ, chi trả cổ tức hoặc tái đầu tư; âm kéo dài có thể làm tăng rủi ro tái cấp vốn.

Lưu ý diễn giải:
- Các ngưỡng tốt/xấu chỉ mang tính tham khảo và cần đặt trong bối cảnh ngành, chu kỳ kinh tế và đặc thù doanh nghiệp.
- Với bài toán tín nhiệm, nên đánh giá đồng thời mức tuyệt đối, xu hướng theo thời gian và độ ổn định của từng chỉ số thay vì nhìn một kỳ đơn lẻ.

Ghi chú logic tài chính quan trọng trong kiểm soát chất lượng dữ liệu:
- Thường áp dụng ràng buộc thứ tự biên lợi nhuận: 
  `gross_profit_margin >= operating_profit_margin >= ebit_margin >= pretax_profit_margin >= net_profit_margin`.

## 3. Lý do gộp từ 22 class ban đầu xuống 3 class

Ban đầu nhãn tín nhiệm ở mức chi tiết (22 notch như `AAA`, `AA+`, ..., `D`). Trong thực nghiệm, việc gộp thành 3 nhóm `IG`, `HY`, `Distressed` được chọn vì các lý do sau:

- Mất cân bằng lớp nghiêm trọng ở 22 lớp:
  Nhiều lớp rất hiếm, làm mô hình khó học ổn định và dễ lệch về lớp phổ biến.

- Giảm nhiễu do khác biệt giữa agency:
  Các mức notch chi tiết dễ bị sai khác theo chuẩn đánh giá của từng tổ chức; gộp nhóm giúp tăng tính nhất quán nhãn.

- Tăng độ tin cậy thống kê:
  Sau khi gộp, mỗi lớp có nhiều mẫu hơn, giúp train/validation ổn định hơn và metric có ý nghĩa hơn.

- Phù hợp mục tiêu quản trị rủi ro:
  Trong nhiều bài toán ứng dụng, quyết định thường ở cấp độ nhóm rủi ro (an toàn, đầu cơ, căng thẳng) thay vì từng notch rất nhỏ.

- Cân bằng giữa độ chi tiết và khả năng học:
  3 lớp vẫn giữ được thứ bậc rủi ro chính nhưng giảm đáng kể độ phức tạp mô hình so với 22 lớp.

Tóm lại, chuyển từ 22 lớp sang 3 lớp là bước đánh đổi có chủ đích để tăng độ bền mô hình, giảm overfitting ở lớp hiếm, và giữ được thông tin rủi ro quan trọng cho bài toán dự báo.

## 4. Cơ sở lý thuyết mô hình và đánh giá

### 4.1 Bản chất bài toán phân loại tín nhiệm theo thứ bậc

Với nhãn mục tiêu gồm `IG`, `HY`, `Distressed`, bài toán không chỉ là phân loại đa lớp thông thường mà còn mang tính thứ bậc rủi ro. Điều này có nghĩa là sai số dự báo cần được diễn giải theo mức độ nghiêm trọng:

- Nhầm `IG` thành `HY` thường ít nghiêm trọng hơn nhầm `IG` thành `Distressed`.
- Nhầm `Distressed` thành `IG` là dạng sai số rủi ro cao trong quản trị tín dụng.

Do đó, ngoài việc tối đa độ chính xác chung, mô hình cần được thiết kế và đánh giá theo hướng ưu tiên giảm sai số ở các lớp rủi ro cao, đặc biệt là lớp `Distressed`.

### 4.2 Khung lý thuyết rủi ro tín dụng từ dữ liệu tài chính

Rủi ro tín nhiệm doanh nghiệp thường được phản ánh qua bốn trụ cột chính:

- Thanh khoản: năng lực đáp ứng nghĩa vụ ngắn hạn (`current_ratio`).
- Đòn bẩy tài chính: mức phụ thuộc vào nợ (`debt_equity_ratio`).
- Hiệu quả sinh lời: năng lực tạo lợi nhuận từ hoạt động và tài sản (`ebit_margin`, `roa`, `roe`).
- Chất lượng dòng tiền: khả năng chuyển hóa lợi nhuận kế toán thành tiền thực (`operating_cashflow_ps`, `free_cashflow_ps`).

Về cơ chế tài chính, khi doanh thu suy giảm hoặc chi phí vốn tăng, nhóm doanh nghiệp có đòn bẩy cao và dòng tiền yếu thường bị suy giảm tín nhiệm nhanh hơn. Vì vậy, mô hình dự báo cần học đồng thời mức tuyệt đối, xu hướng biến động và độ ổn định của từng chỉ tiêu qua thời gian.

### 4.3 Lý thuyết xử lý mất cân bằng lớp

Mất cân bằng lớp gây ra hiện tượng mô hình thiên lệch về lớp phổ biến và giảm độ nhạy ở lớp hiếm. Các hướng xử lý phổ biến gồm:

- Điều chỉnh hàm mất mát theo trọng số lớp (class-weight).
- Tăng cường mẫu lớp hiếm bằng tái lấy mẫu hoặc sinh mẫu tổng hợp.
- Kết hợp chiến lược dữ liệu và chiến lược mô hình để cân bằng giữa độ bao phủ và độ chính xác.

Trong đề tài này, việc gộp từ 22 lớp xuống 3 lớp là bước nền để ổn định thống kê. Sau đó có thể bổ sung sinh mẫu có kiểm soát cho lớp `Distressed`, nhưng phải giám sát chặt rủi ro sai lệch phân phối và nhiễu do mẫu tổng hợp.

### 4.4 Cơ sở chuỗi thời gian và nguyên tắc chống rò rỉ dữ liệu

Với dữ liệu tín nhiệm theo thời gian, việc tách train/validation/test phải tuân thủ trật tự thời gian để tránh thiên lệch lạc quan. Nguyên tắc cốt lõi là point-in-time:

- Mọi đặc trưng tại thời điểm $t$ chỉ được tính từ thông tin có trước hoặc bằng $t$.
- Không sử dụng thông tin tương lai của cùng doanh nghiệp hoặc của toàn bộ tập dữ liệu khi tạo đặc trưng.
- Ưu tiên đánh giá bằng walk-forward để mô phỏng đúng bối cảnh triển khai thực tế.

Nếu vi phạm các nguyên tắc này, mô hình có thể đạt điểm số cao trong thử nghiệm nhưng giảm mạnh khi triển khai thực tế.

### 4.5 Hệ chỉ tiêu đánh giá phù hợp bài toán tín nhiệm

Với dữ liệu mất cân bằng và lớp mục tiêu có thứ bậc, chỉ dùng accuracy là chưa đủ. Cần kết hợp các chỉ số sau:

- Macro-F1: đánh giá cân bằng chất lượng dự báo giữa các lớp.
- Balanced Accuracy: giảm thiên lệch do chênh lệch số lượng mẫu giữa các lớp.
- Recall lớp `Distressed`: phản ánh khả năng phát hiện nhóm rủi ro cao.
- Ma trận nhầm lẫn: phân tích hướng sai số giữa các mức rủi ro.
- Chỉ số hiệu chỉnh xác suất (ví dụ Brier score): đánh giá độ tin cậy của xác suất dự báo.

Việc theo dõi đồng thời các chỉ số này giúp cân bằng giữa hiệu năng tổng thể và yêu cầu an toàn trong quyết định tín dụng.

### 4.6 Kiểm định độ tin cậy và tính bền của kết quả

Để kết quả có giá trị học thuật và ứng dụng, cần bổ sung các kiểm định sau:

- Ablation study: loại bỏ từng nhóm đặc trưng để đo đóng góp biên.
- Sensitivity analysis: kiểm tra độ ổn định theo từng giai đoạn thời gian hoặc từng ngành.
- Khoảng tin cậy cho metric (bootstrap hoặc lặp theo nhiều seed): giảm rủi ro kết luận từ một lần chạy đơn lẻ.

Các kiểm định này giúp chứng minh mô hình không chỉ tốt ở một cấu hình cụ thể mà còn có khả năng tổng quát hóa.

### 4.7 Hàm ý triển khai trong quản trị rủi ro

Trong thực tế, đầu ra mô hình nên được chuyển thành tín hiệu hỗ trợ quyết định:

- Cảnh báo sớm doanh nghiệp có xác suất cao rơi vào `Distressed`.
- Xếp hạng mức ưu tiên tái thẩm định theo xác suất và mức độ bất định của dự báo.
- Kết hợp đầu ra mô hình với quy tắc chuyên gia để tạo quy trình kiểm soát rủi ro lai.

Theo cách tiếp cận này, mô hình đóng vai trò hệ thống cảnh báo định lượng, hỗ trợ chuyên gia tín dụng ra quyết định nhất quán và kịp thời hơn.

