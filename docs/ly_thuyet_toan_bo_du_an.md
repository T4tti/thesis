# Lý thuyết tổng quan toàn bộ dự án

## 1. Bối cảnh và mục tiêu nghiên cứu

Dự án tập trung vào bài toán **xếp hạng tín nhiệm doanh nghiệp (corporate credit rating)** dựa trên dữ liệu tài chính theo thời gian.

Mục tiêu chính:
- Xây dựng pipeline tái lập từ thu thập dữ liệu đến huấn luyện mô hình.
- Chuẩn hóa dữ liệu từ nhiều nguồn khác nhau về cùng schema.
- Giảm tác động của mất cân bằng lớp (class imbalance).
- Cải thiện dự báo theo bản chất thứ bậc (ordinal) của nhãn tín nhiệm.
- Giảm hiện tượng mô hình chỉ học theo quán tính nhãn cũ (persistence bias).

Bài toán phân lớp bao gồm nhiều mức xếp hạng từ `D` đến `AAA` (thang chi tiết theo `rating_detail`, hoặc thang gộp theo `rating_class`, hoặc nhị phân `binary_rating`).

## 2. Kiến trúc tổng thể của project

Project được tổ chức theo hướng script hóa, tái lập, và tách rõ trách nhiệm:

- `src/scrapers/`: Thu thập dữ liệu ngoài hệ thống.
- `src/pipelines/`: ETL, chuẩn hóa, augmentation, benchmark.
- `src/models/`: Kiến trúc mô hình và utility huấn luyện.
- `data/raw/`: Dữ liệu đầu vào gốc.
- `data/processed/`: Dữ liệu sau xử lý/augmentation.
- `data/reports/`: Báo cáo benchmark, augmentation.
- `artifacts/models/`, `credit_rating_artifacts/`: mô hình và artifact huấn luyện.
- `notebooks/`: Thử nghiệm và huấn luyện theo workflow nghiên cứu.

## 3. Nguồn dữ liệu và chuẩn hóa dữ liệu

## 3.1 Nguồn dữ liệu

Dự án dùng các nguồn chính:
- Dữ liệu scraped từ FiinRatings (`src/scrapers/fiinratings_scraper.py`).
- Dataset `corporate_rating.csv`.
- Dataset `corporateCreditRatingWithFinancialRatios.csv`.

Ngoài ra có các tập dữ liệu tăng cường (SMOTE, CTGAN, TabDDPM, TimeGAN) trong `data/processed/`.

## 3.2 Thu thập dữ liệu FiinRatings

Script `src/scrapers/fiinratings_scraper.py` thực hiện:
- Gọi AJAX endpoint `https://fiinratings.vn/vi/ratings/indexajax`.
- Parse HTML dạng fragment `<tr>` bằng cách bọc tạm vào `<table><tbody>...</tbody></table>`.
- Duy trì pacing để tránh quá tải server:
  - `DELAY_PAGE = 1.5`
  - `DELAY_PDF = 0.8`
- Lưu CSV bằng `utf-8-sig` để tương thích tiếng Việt.
- Tải báo cáo PDF về `data/external/fiinratings_output/pdfs/`.

Đây là bước ingestion quan trọng để mở rộng nguồn dữ liệu thực tế.

## 3.3 Hợp nhất hai bộ dữ liệu tài chính

Script `src/pipelines/merge_dataset.py` triển khai pipeline chuẩn hóa và gộp dữ liệu theo nhiều bước minh bạch:

1. Chuẩn hóa tên cột giữa hai file nguồn.
2. Chuẩn hóa nhãn tín nhiệm:
   - Map `rating_detail` sang `rating_class` theo nhóm lớn (`AAA`, `AA`, ..., `D`).
3. Chuẩn hóa sector giữa các quy ước đặt tên khác nhau.
4. Chuẩn hóa đơn vị tỷ lệ tài chính:
   - Nhiều cột phần trăm từ file 2 được chia `100` để đồng nhất về dạng decimal.
5. Chuẩn hóa thời gian `rating_date`.
6. Gộp dữ liệu bằng `concat` theo dòng.
7. Loại trùng theo khóa:
   - `ticker`, `rating_agency`, `rating_date`.
   - Ưu tiên bản ghi từ nguồn `credit_rating_financial`.
8. Tái tạo `binary_rating` từ `rating_class` (investment grade vs speculative grade).
9. Winsorize outlier tại ngưỡng 1% - 99% cho tập cột tài chính.
10. Xuất 2 bộ:
   - `merged_credit_rating_full.csv` (giữ nhiều cột nhất).
   - `merged_credit_rating_common.csv` (bộ cột chung dùng cho modeling).

Ý nghĩa lý thuyết: bước này làm đồng nhất ngữ nghĩa dữ liệu và giảm sai lệch hệ đo trước khi học máy.

## 3.4 Ý nghĩa các thuộc tính trong bộ dữ liệu 3 nhóm

Với file `data/processed/merged_credit_rating_common_3groups.csv`, các thuộc tính mang ý nghĩa như sau.

### Nhóm định danh và ngữ cảnh

| Thuộc tính | Kiểu dữ liệu | Ý nghĩa nghiệp vụ | Ghi chú dùng mô hình |
|---|---|---|---|
| `rating_detail` | Categorical | Nhãn mục tiêu 3 nhóm rủi ro tín nhiệm: `IG`, `HY`, `Distressed`. | Đây là target trong thiết lập 3 lớp (ordinal thô theo mức rủi ro tăng dần: `IG -> HY -> Distressed`). |
| `company_name` | Text | Tên doanh nghiệp được xếp hạng. | Chủ yếu để truy vết, thường không dùng trực tiếp làm feature vì dễ gây high cardinality và leakage ngữ danh. |
| `ticker` | Categorical | Mã chứng khoán/doanh nghiệp. | Quan trọng cho grouping theo thực thể (windowing, delta theo thời gian, split chống leakage theo entity). |
| `rating_agency` | Categorical | Tổ chức đánh giá tín nhiệm (Moody's, S&P, Fitch, EJR, ...). | Có thể chứa khác biệt phong cách chấm điểm giữa agency; thường encode để mô hình hấp thụ hiệu ứng hệ thống. |
| `rating_date` | Date | Ngày công bố xếp hạng. | Dùng tạo đặc trưng thời gian (`year`, `month`, `quarter`) và sắp chuỗi theo `ticker`. |
| `sector` | Categorical | Ngành hoạt động của doanh nghiệp. | Dùng để chuẩn hóa theo ngành (sector-relative z-score), tránh so sánh chéo ngành không công bằng. |
| `source` | Categorical | Nguồn bản ghi sau hợp nhất (`corporate_rating` hoặc `credit_rating_financial`). | Dùng cho kiểm soát domain shift giữa nguồn dữ liệu. |

### Nhóm thanh khoản, đòn bẩy, hiệu quả và sinh lời

| Thuộc tính | Ý nghĩa tài chính | Diễn giải rủi ro tín nhiệm (xu hướng chung) |
|---|---|---|
| `current_ratio` | Tài sản ngắn hạn / nợ ngắn hạn, phản ánh khả năng thanh toán ngắn hạn. | Tăng vừa phải thường giảm rủi ro vỡ nợ ngắn hạn; quá thấp là tín hiệu căng thẳng thanh khoản. |
| `debt_equity_ratio` | Tổng nợ / vốn chủ sở hữu, phản ánh mức đòn bẩy tài chính. | Cao kéo dài thường làm tăng rủi ro tín nhiệm; giá trị âm có thể xuất hiện khi vốn chủ âm, là cảnh báo cấu trúc vốn bất thường. |
| `asset_turnover` | Doanh thu / tổng tài sản, phản ánh hiệu quả sử dụng tài sản. | Cao hơn thường tích cực về hiệu quả vận hành, nhưng cần diễn giải theo ngành (biên lợi nhuận thấp có thể vẫn turnover cao). |
| `roe` | Lợi nhuận ròng / vốn chủ sở hữu. | ROE dương, ổn định thường tích cực; ROE âm hoặc biến động mạnh là tín hiệu suy giảm chất lượng tín dụng. |
| `roa` | Lợi nhuận ròng / tổng tài sản. | ROA thấp hoặc âm kéo dài thường liên quan chất lượng tài sản và khả năng sinh lời yếu. |

### Nhóm biên lợi nhuận theo tầng lợi nhuận

| Thuộc tính | Ý nghĩa tài chính | Diễn giải rủi ro tín nhiệm (xu hướng chung) |
|---|---|---|
| `gross_profit_margin` | Biên lợi nhuận gộp. | Cao và ổn định cho thấy năng lực định giá/sản xuất tốt, hỗ trợ khả năng trả nợ. |
| `operating_profit_margin` | Biên lợi nhuận hoạt động. | Đo hiệu quả cốt lõi sau chi phí vận hành; giảm mạnh là tín hiệu suy yếu hoạt động. |
| `ebit_margin` | Biên EBIT (lợi nhuận trước lãi vay và thuế). | Quan trọng với tín dụng vì gắn trực tiếp với năng lực chi trả lãi vay. |
| `pretax_profit_margin` | Biên lợi nhuận trước thuế. | Phản ánh hiệu quả trước tác động thuế; giảm sâu thường đi kèm rủi ro tài chính tăng. |
| `net_profit_margin` | Biên lợi nhuận ròng. | Kết quả cuối cùng sau mọi chi phí; âm kéo dài liên quan khả năng suy giảm tín nhiệm. |

Lưu ý quan trọng về logic tài chính: trong nhiều pipeline augment (đặc biệt CTGAN constraints), thứ tự biên lợi nhuận được ràng buộc theo nguyên tắc:

`gross_profit_margin >= operating_profit_margin >= ebit_margin >= pretax_profit_margin >= net_profit_margin`

Điều này nhằm giữ tính hợp lệ kế toán của dữ liệu tổng hợp.

### Nhóm dòng tiền trên mỗi cổ phần

| Thuộc tính | Ý nghĩa tài chính | Diễn giải rủi ro tín nhiệm (xu hướng chung) |
|---|---|---|
| `operating_cashflow_ps` | Dòng tiền từ hoạt động kinh doanh trên mỗi cổ phần. | Dương và ổn định cho thấy doanh nghiệp tạo tiền tốt từ hoạt động lõi, thường hỗ trợ chất lượng tín dụng. |
| `free_cashflow_ps` | Dòng tiền tự do trên mỗi cổ phần sau đầu tư duy trì. | Dương bền vững hỗ trợ trả nợ và chống sốc chu kỳ; âm kéo dài có thể tăng rủi ro tái cấp vốn. |

### Ý nghĩa nhãn 3 nhóm trong thực nghiệm

Trong file 3 nhóm, cột `rating_detail` không còn là thang notch chi tiết (`AAA`, `AA+`, ...), mà đã quy về 3 mức:

- `IG` (Investment Grade): nhóm chất lượng tín dụng tốt, rủi ro thấp hơn.
- `HY` (High Yield): nhóm đầu cơ, rủi ro cao hơn, lợi suất yêu cầu thường cao hơn.
- `Distressed`: nhóm căng thẳng tài chính cao, xác suất suy giảm tín nhiệm/vỡ nợ lớn hơn.

Vì vậy, khi huấn luyện mô hình trên file này, cần coi `rating_detail` là biến phân lớp 3 nhóm và cập nhật lại mọi phần encode/metric theo đúng ngữ nghĩa mới.

## 4. Tối ưu đặc trưng và xử lý dữ liệu trước học

Script `src/pipelines/optimize_dataset.py` tạo phiên bản tối ưu từ dữ liệu augment TimeGAN:
- Input mặc định: `data/processed/train_augmented_timegan.csv`.
- Output: `data/processed/train_augmented_timegan_optimized.csv`.

Các phép biến đổi chính:
- **Sector-relative normalization**: z-score theo từng ngành.
- **Temporal delta features**: thêm cột `lag1` và `delta` theo `ticker` + `rating_date`.
- **Robust outlier capping**: IQR clipping với hệ số 3.0.
- **Dọn lag trung gian**: loại cột `_lag1` sau khi tạo delta.

Ý nghĩa lý thuyết:
- Chuẩn hóa theo ngành giúp mô hình so sánh công bằng giữa các doanh nghiệp khác lĩnh vực.
- Delta feature giúp mô hình học biến động thay vì chỉ học mức tuyệt đối.

## 5. Tăng cường dữ liệu (augmentation)

## 5.1 TS-SMOTE cho dữ liệu chuỗi panel

Script `src/pipelines/smote_augment.py` triển khai TS-SMOTE theo hướng giữ cấu trúc thời gian:
- Split train/val/test mặc định 80/10/10.
- Mã hóa ordinal cho `rating_detail` theo thang 22 lớp từ `D` đến `AAA`.
- Stage 1: bootstrap bằng `RandomOverSampler` cho lớp cực hiếm.
- Stage 2: nội suy theo cặp thời điểm gần kề trong cùng `ticker` (ưu tiên), fallback theo láng giềng cùng lớp/same sector.

Output chính:
- `data/processed/train_smote_augmented.csv`
- `data/reports/smote_augmentation_report.csv`
- Kèm `val_split.csv`, `test_split.csv`.

Lý thuyết cốt lõi: khác với SMOTE thường, TS-SMOTE cố giữ tính liên tục temporal để tránh mẫu tổng hợp vô lý theo chuỗi thời gian.

## 5.2 CTGAN constraints và tính hợp lệ tài chính

Module `src/pipelines/ctgan_constraints.py` định nghĩa:
- Hard bounds tài chính (`FINANCIAL_BOUNDS`).
- Chuẩn hóa nhất quán `rating_class` và `binary_rating`.
- Ràng buộc thứ tự biên lợi nhuận:
  - `gross_profit_margin >= operating_profit_margin >= ebit_margin >= pretax_profit_margin >= net_profit_margin`.
- Constraint CAG của SDV:
  - `FixedCombinations`
  - `Inequality`

Mục tiêu lý thuyết: dữ liệu sinh bởi CTGAN không chỉ giống phân phối, mà còn tôn trọng logic kế toán/tài chính cơ bản.

## 5.3 TabDDPM và TimeGAN (theo tài liệu kế hoạch)

Hai tài liệu:
- `docs/tabddpm_data_split_and_preprocess_plan.md`
- `docs/ctgan_data_split_and_preprocess_plan.md`

Đặt ra nguyên tắc chung:
- Fit preprocessing trên train, tuyệt đối tránh leakage từ val/test.
- Chỉ augmentation trên train.
- So sánh công bằng giữa phương pháp bằng cùng protocol split/preprocess/eval.

Tài liệu `TimeGAN.md` nêu rationale chuyển từ CTGAN tabular sang TimeGAN cho dữ liệu có phụ thuộc thời gian.

## 6. Benchmark và chọn chiến lược mục tiêu

## 6.1 Benchmark target labels

Script `src/pipelines/benchmark_targets.py` so sánh ba nhãn:
- `binary_rating`
- `rating_class`
- `rating_detail`

Thiết kế benchmark:
- Feature table nhất quán.
- Pipeline mixed-type:
  - Numeric: median imputation.
  - Categorical: most-frequent + OneHotEncoder.
- Classifier nhanh: `SGDClassifier(loss='log_loss')`.
- Tùy chọn benchmark ordinal với `Ridge` dự báo rank.

Metrics gồm:
- `accuracy`, `balanced_accuracy`, `f1_macro`, `f1_weighted`.
- Với ordinal: `mae_rank`, `qwk`, `within_1_notch`, thống kê lệch notch.

Output: `data/reports/benchmark_targets_report.csv`.

## 6.2 Benchmark giữa các tập augment train

Script `src/pipelines/benchmark_augmented_trains.py` benchmark các tập:
- CTGAN
- TabDDPM
- SMOTE
- TimeGAN

Thiết kế:
- Repeated Stratified K-Fold.
- Dùng chung pipeline feature/classifier từ benchmark target.
- Report cả mean và std cho metric.

Output: `data/reports/benchmark_augmented_train_report.csv`.

Ý nghĩa: giúp chọn phương pháp augmentation tối ưu theo hiệu quả thực nghiệm, không chọn theo cảm tính.

## 7. Lý thuyết mô hình học máy chính

## 7.1 TLSTM-Fuzzy V2

Module `src/models/tlstm_fuzzy_v2.py` là gói cải tiến Priority 1 cho mô hình Transformer-BiLSTM-Fuzzy.

Thành phần chính:
- `FocalOrdinalLossV2`:
  - Focal CE + regularization theo khoảng cách ordinal.
  - Hỗ trợ class weights (inverse frequency / class-balanced).
- `GatedContextInjection`:
  - Thay concat trực tiếp `last_y` bằng cơ chế gated cross-attention để giảm shortcut learning.
- `FuzzyLayer`:
  - Gaussian membership expansion cho từng feature.

Lý thuyết:
- Credit rating có thứ bậc, nên loss ordinal hợp lý hơn cross-entropy thuần.
- Context `last_y` vừa hữu ích vừa nguy hiểm; cơ chế gate giúp kiểm soát mức phụ thuộc.

## 7.2 TLSTM-Fuzzy V3

Module `src/models/tlstm_fuzzy_v3.py` bổ sung Priority 2 + 3:
- **RoPE** (`RotaryPositionalEmbedding`) trong self-attention thời gian.
- **MultiScalePool**: tổng hợp ngữ cảnh theo nhiều tỷ lệ (attentive + last + mean).
- **FeatureInteractionLayer**: bắt tương tác chéo giữa chỉ số tài chính.
- **CORN ordinal loss** (`CornOrdinalLoss`) cho bài toán ordinal nhiều lớp.
- Utilities huấn luyện nâng cao:
  - SWA.
  - Cosine warm restarts scheduler.

Lý thuyết cốt lõi:
- RoPE giúp mô hình hóa vị trí tương đối trong chuỗi tốt hơn.
- CORN chuyển bài toán K lớp ordinal thành K-1 bài toán nhị phân có điều kiện, phù hợp bản chất xếp hạng tín nhiệm.

## 7.3 Anti-persistence training utilities

Module `src/models/training_utils.py` mô tả trực diện nguyên nhân và cách xử lý persistence bias.

Các thành phần:
- `ContextScheduler`:
  - Phase 1: mask/permute mạnh `last_y` để buộc học từ financial features.
  - Phase 2: giảm dần masking.
  - Phase 3: fine-tune masking thấp.
- `TransitionScheduler`:
  - Bật transition penalty sớm (không warmup quá trễ).
- `build_change_weighted_sampler`:
  - Tăng trọng số cho mẫu có thay đổi nhãn.
- `compute_transition_penalty_v2`:
  - Penalty theo chênh lệch ordinal giữa nhãn thật và nhãn trước đó.

Mục tiêu thực nghiệm trong module:
- Tăng `ChgAcc` (Change Accuracy).
- Vượt persistence baseline về `F1-weighted`.

## 7.4 HHGNN-Fuzzy

Module `src/models/hhgnn_fuzzy.py` và pipeline `src/pipelines/train_hhgnn.py` tạo tuyến đồ thị cho credit rating.

Lý thuyết kiến trúc:
- Mỗi doanh nghiệp được biểu diễn thành đồ thị đặc trưng (feature graph):
  - Node: chỉ số tài chính.
  - Edge weight: fuzzy membership từ cosine similarity.
- Mô hình `HHGNNFuzzyClassifier` dùng nhiều tầng `GATConv` + pooling (`global_mean_pool`, `global_max_pool`) + head phân loại.
- Loss: `FuzzyFocalLoss` có thể kết hợp class weight và fuzzy sample weight.

Chi tiết quan trọng trong chuẩn bị dữ liệu:
- `prepare_static_company_dataset` chuẩn hóa theo train.
- Với HHGNN static, giữ bản ghi gần nhất theo mỗi `ticker`.
- Có pipeline huấn luyện script hóa, lưu đồ thị metric/loss/confusion matrix, hỗ trợ SHAP tùy chọn.

## 8. Đánh giá mô hình và chỉ số

Các nhóm metric xuất hiện xuyên suốt pipeline:
- Accuracy.
- Balanced Accuracy.
- F1 macro.
- F1 weighted.
- MAE theo rank ordinal.
- QWK (quadratic weighted kappa) cho bài toán thứ bậc.
- Chỉ số liên quan persistence/transition trong utility huấn luyện (`ChgAcc`, stay/change behavior).

Thông điệp lý thuyết:
- Không nên đánh giá mỗi accuracy khi dữ liệu mất cân bằng và có trật tự nhãn.
- Cần kết hợp metric phân lớp + metric ordinal + metric nhạy với thay đổi nhãn theo thời gian.

## 9. Tính tái lập và artifact

Project ưu tiên tái lập qua:
- Script entrypoint rõ ràng (`if __name__ == "__main__"`).
- Đường dẫn output ổn định trong `data/processed/`, `data/reports/`, `artifacts/models/`.
- Seed cố định trong các pipeline benchmark/train.
- Báo cáo CSV/plot/config được lưu sau mỗi tiến trình.

Script hỗ trợ trực quan tổng quan:
- `src/pipelines/generate_overview_diagram.py` tạo hình minh họa luồng train/test cho báo cáo.

## 10. Hệ phụ thuộc chính

Từ `requirements.txt`:
- `requests`, `beautifulsoup4`, `pandas`, `lxml`, `openpyxl`
- `scikit-learn`, `imbalanced-learn`
- `coral-pytorch`

Lưu ý với HHGNN:
- Cần thêm `torch-geometric` và các gói phụ thuộc tương ứng môi trường CUDA/CPU.

## 11. Vai trò của notebook trong project

Các notebook trong `notebooks/` là nơi:
- Thử nghiệm kiến trúc (Transformer-LSTM, CORN, PatchTST, HHGNN).
- Chạy ablation nhanh.
- Trực quan hóa kết quả.

Trong khi đó, phần `src/` đóng vai trò workflow chính thức để tái lập.

## 12. Kết luận lý thuyết

Dự án xây dựng một hệ thống dự báo xếp hạng tín nhiệm theo hướng:
- Dữ liệu chuẩn hóa đa nguồn.
- Tăng cường dữ liệu có kiểm soát bằng nhiều phương pháp.
- Mô hình hóa theo bản chất ordinal và temporal.
- Kiểm soát persistence bias có chủ đích trong huấn luyện.
- Bổ sung nhánh đồ thị mờ (HHGNN-Fuzzy) để mở rộng năng lực biểu diễn quan hệ phi tuyến.

Về mặt phương pháp luận, đóng góp lớn nhất của project nằm ở việc kết hợp:
- Feature engineering theo ngành + thời gian.
- Ordinal-aware learning (FocalOrdinal/CORN).
- Anti-shortcut curriculum cho tín hiệu `last_y`.
- Augmentation benchmark hóa trên cùng protocol.

Nhờ đó, project vừa phục vụ nghiên cứu học thuật, vừa đủ cấu trúc để chuyển thành pipeline sản xuất thử nghiệm có kiểm soát.
