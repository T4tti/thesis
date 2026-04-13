import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

# 1. Load data
df = pd.read_csv('data/processed/merged_credit_rating_common_3groups.csv')

# 2. Identify features and target
target = 'rating_detail'
features = [
    'current_ratio', 'debt_equity_ratio', 'gross_profit_margin', 
    'operating_profit_margin', 'ebit_margin', 'pretax_profit_margin', 
    'net_profit_margin', 'asset_turnover', 'roe', 'roa', 
    'operating_cashflow_ps', 'free_cashflow_ps'
]

X = df[features]
y = df[target]

# Feature descriptions với chuẩn nghiệp vụ xếp hạng tín nhiệm cao cấp (S&P, Moody's, FiinRatings)
desc = {
    'current_ratio': 'Năng lực thanh khoản ngắn hạn (Short-term Liquidity). Đánh giá "tấm đệm" lưu chuyển tiền tệ (cash flow buffer) để chống chịu các cú sốc thanh khoản trong 12 tháng tới. Tỷ lệ < 1.0 (hoặc thấp so với trung bình ngành) báo hiệu rủi ro tái cấp vốn (refinancing risk) khẩn cấp.',
    'debt_equity_ratio': 'Rủi ro cơ cấu vốn & Đòn bẩy tài chính (Financial Leverage). Là một trong những "hard-constraint" (chốt chặn) quan trọng nhất. Đòn bẩy cao phản ánh lớp vốn đệm (equity cushion) mỏng, làm tăng theo cấp số nhân Xác suất vỡ nợ (Probability of Default - PD) khi vĩ mô suy thoái.',
    'gross_profit_margin': 'Lợi thế cạnh tranh & Gia tăng giá trị (Pricing Power). Là chỉ báo cốt lõi của khả năng chuyển dịch chi phí đầu vào lên người tiêu dùng. Biên gộp càng ổn định qua các chu kỳ kinh tế chứng tỏ rào cản gia nhập ngành lớn, giúp cải thiện Điểm rủi ro kinh doanh (Business Risk Profile).',
    'operating_profit_margin': 'Hiệu quả vận hành (Operational Efficiency). Trong thẻ điểm tín nhiệm (Credit Scorecard), biên hoạt động đánh giá sức khỏe sinh lời thuần túy từ mảng kinh doanh lõi, loại bỏ các yếu tố thu nhập bất thường. Sự ổn định của chỉ số này ảnh hưởng mạnh tới xếp loại Ổn định (Outlook Stable).',
    'ebit_margin': 'Khả năng sinh lời lõi phục vụ trả nợ (Debt Servicing Base). EBIT là nguồn thu vô cùng quan trọng dùng để tính EBIT Interest Coverage (Khả năng thanh toán lãi vay). Biên EBIT càng dày chứng tỏ bộ đệm sẵn sàng thanh toán nghĩa vụ nợ cố định (Fixed-charge cover) càng vững mạnh.',
    'pretax_profit_margin': 'Mức sinh lời trước mức thuế phải nộp. Hữu ích cho nghiệp vụ so sánh tính điểm tín nhiệm ngang hàng (Peer comparison) trong các ngành có lá chắn thuế (Tax shield) biến động mạnh giữa các quốc gia hoặc chu kỳ kinh tế.',
    'net_profit_margin': 'Hiệu quả tạo vốn tự có (Capital Generation Capacity). Biên ròng quyết định dòng lợi nhuận giữ lại (Retained Earnings) dùng để bổ sung trực tiếp vào Vốn chủ sở hữu, từ đó cấu trúc lại sức khỏe của bảng cân đối kế toán. Giảm sút sẽ cảnh báo cạn kiệt vốn (Capital Depletion).',
    'asset_turnover': 'Hiệu suất sử dụng vốn (Asset Efficiency/Capital Intensity). Đặc biệt cực kỳ quan trọng với công nghiệp nặng/bán lẻ. Vòng quay sụt giảm là "Red Flag" về hàng tồn kho/khoản phải thu ứ đọng, báo trước dòng tiền HĐKD sẽ suy yếu.',
    'roe': 'Lợi nhuận cổ đông (Return on Equity). Trong nghiệp vụ Credit Rating, ROE quá cao có thể là hệ quả của việc lạm dụng nợ vay (hiệu ứng DuPont). Cơ quan xếp hạng phân tích chéo ROE với Đòn bẩy để nhận diện các thủ thuật tối ưu hóa chỉ số che bảo rủi ro vỡ nợ.',
    'roa': 'Sinh lời tổng thể (Return on Assets). Đánh giá chân thực năng lực quản trị vốn (Management Capability) trên mọi đồng tài trợ (Dù từ Nợ hay Vốn) mà không bị che lấp / bóp méo bởi hiệu ứng đòn bẩy tài chính.',
    'operating_cashflow_ps': 'Sức khỏe thực chất. Dòng tiền HĐKD (CFO) là chỉ báo Sinh tồn (Cash is King). CFO có thể phát hiện thủ thuật kế toán bóp méo lợi nhuận ghi sổ (Earnings Management). Dòng tiền CFO âm kéo dài trực tiếp dẫn đến rủi ro vỡ nợ kỹ thuật (Technical Default) dù có lãi.',
    'free_cashflow_ps': 'Tự chủ tài chính (Stand-alone Credit Profile). FCF (Dòng tiền tự do) đo lường nguồn thặng dư sau chi phí đầu tư duy trì (CapEx). FCF dồi dào chứng tỏ khả năng tự trả nền gốc nợ mà không phải đảo nợ (Roll-over). FCF âm thường là đặc trưng của khu vực doanh nghiệp trái phiếu rác (Junk / Speculative Grade).'
}

# 3. Preprocess
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)
X_imputed = pd.DataFrame(X_imputed, columns=features)

le = LabelEncoder()
y_encoded = le.fit_transform(y) # e.g. IG, HY...

# 4. Train Model
model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
model.fit(X_imputed, y_encoded)

# 5. Extract Feature Importances (Gini importance)
importances = model.feature_importances_

# Create a dataframe for features
feat_df = pd.DataFrame({
    'Trường': features,
    'Importance': importances
})
feat_df = feat_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)
feat_df['Rank'] = feat_df.index + 1

# Generate the final output
out_data = []

# Title row
out_data.append({
    'TRƯỜNG DỮ LIỆU – Chuẩn nghiệp vụ xếp hạng doanh nghiệp (S&P/Moody\'s/FiinRatings)': 'Tập dữ liệu Corporate Credit Rating. Hệ thống được mapping với các khái niệm lõi định lượng rủi ro vỡ nợ, thẻ điểm tín nhiệm (Credit Scorecard) và đánh giá sức bền tài chính (Financial Resiliency).',
    'Kiểu dữ liệu': '',
    'Mô tả phân tích (Chuẩn nghiệp vụ tín nhiệm)': '',
    'Góc nhìn Rủi ro (Risk Impact)': '',
    'Đóng góp (%) / Trọng số (Gini)': '',
    'Phân lớp Trọng yếu': '',
    'Insights (Dành cho Transformer-BiLSTM)': ''
})

# Header row
out_data.append({
    'TRƯỜNG DỮ LIỆU – Chuẩn nghiệp vụ xếp hạng doanh nghiệp (S&P/Moody\'s/FiinRatings)': 'Tên trường (Feature)',
    'Kiểu dữ liệu': 'Kiểu dữ liệu',
    'Mô tả phân tích (Chuẩn nghiệp vụ tín nhiệm)': 'Ý nghĩa trong Thẻ điểm Xếp hạng Tín nhiệm (Credit Scorecard)',
    'Góc nhìn Rủi ro (Risk Impact)': 'Tác động đến PD (Probability of Default)',
    'Đóng góp (%) / Trọng số (Gini)': 'Scorecard \nWeight (\ntheo Mô hình)',
    'Phân lớp Trọng yếu': 'Ranking / \nMức độ cốt lõi',
    'Insights (Dành cho Transformer-BiLSTM)': 'Ghi chú cho Kiến trúc Modeling'
})

# Metadata columns
metadata_cols = [
    ('rating_detail', 'String (Categorical)', 'Nhãn mục tiêu: Thuộc các cấp độ Non-Investment Grade (HY), Investment Grade (IG) theo khung Master Scale nghiệp vụ chuẩn.', 'Biến học tập lõi cho hàm CORN (Conditional Ordinal Regression)'),
    ('company_name', 'String', 'Tên tổ chức phát hành (Issuer). Định danh thực thể độc lập phân tích.', 'Tránh rỉ rỉ thông tin trong chia tập Train/Test theo chuỗi'),
    ('ticker', 'String', 'Mã chứng khoán / Mã tham chiếu', 'Định danh chuỗi thời gian của entity'),
    ('rating_agency', 'String', 'Cơ quan cấp hạng. Yếu tố quan trọng để chuẩn hóa sự chênh lệch (notch divergence) giữa các phương pháp luật của từng Agency.', 'Có thể đóng vai trò làm Condition Feature/Embedding cho mô hình'),
    ('rating_date', 'Timeline Feature', 'Ngày cấp hạng. Mốc thời gian kích hoạt các cảnh báo/theo dõi.', 'Trục X (Temporal Axis) cho mô hình BiLSTM, chuỗi sliding window'),
    ('sector', 'String (Categorical)', 'Nhóm ngành hoạt động (Business Sector). Xử lý rủi ro ngành (Industry Risk) trong khung rủi ro kinh doanh.', 'Phân tán tuyệt đối độ tương đương rủi ro: 1 tỷ nợ của BĐS khác 1 tỷ nợ của Bán lẻ.'),
    ('source', 'String', 'Nguồn thu thập dữ liệu (Refinitiv, Bloomberg, FiinPro...)', 'Hỗ trợ audit xuất xứ số liệu')
]
for col, dtype, desc_t, example in metadata_cols:
    out_data.append({
        'TRƯỜNG DỮ LIỆU – Chuẩn nghiệp vụ xếp hạng doanh nghiệp (S&P/Moody\'s/FiinRatings)': col,
        'Kiểu dữ liệu': dtype,
        'Mô tả phân tích (Chuẩn nghiệp vụ tín nhiệm)': desc_t,
        'Góc nhìn Rủi ro (Risk Impact)': 'Dữ liệu điều hướng / Meta',
        'Đóng góp (%) / Trọng số (Gini)': 'N/A',
        'Phân lớp Trọng yếu': 'Metadata',
        'Insights (Dành cho Transformer-BiLSTM)': example
    })

# Feature columns
for i, row in feat_df.iterrows():
    f = row['Trường']
    imp = row['Importance']
    rank = row['Rank']
    
    impact = '...'
    if 'ratio' in f or f in ['roe', 'roa', 'operating_cashflow_ps', 'free_cashflow_ps'] or 'margin' in f:
        impact = 'Năng lực tài chính MẠNH -> Tương quan dương với mức đánh giá Đầu tư (IG). Giảm mạnh Nguy cơ vỡ nợ gốc/lãi.'
    if 'debt' in f:
        impact = 'Gia tăng Áp lực Đòn bẩy. Nếu tỷ lệ này bung rộng, báo hiệu cấu trúc vốn mất cân đối nghiêm trọng, kích hoạt tín hiệu hạ bậc (Downgrade).'
        
    out_data.append({
        'TRƯỜNG DỮ LIỆU – Chuẩn nghiệp vụ xếp hạng doanh nghiệp (S&P/Moody\'s/FiinRatings)': f,
        'Kiểu dữ liệu': 'Financial Ratio',
        'Mô tả phân tích (Chuẩn nghiệp vụ tín nhiệm)': desc.get(f, ''),
        'Góc nhìn Rủi ro (Risk Impact)': impact,
        'Đóng góp (%) / Trọng số (Gini)': f"{imp*100:.1f}%",
        'Phân lớp Trọng yếu': f"Rank {rank} / {len(features)}",
        'Insights (Dành cho Transformer-BiLSTM)': f'{"Top Critical (Chốt chặn): LSTM cần trích xuất ngay lập tức các độ bất thường của chỉ số này so với trung bình nhóm (Sector-outliers)." if rank <=3 else ("High Importance: Là căn cứ để xây dựng nhánh Cross-Feature trong mô hình." if rank <=7 else "Supporting Metric: Vai trò làm nhiễu hoặc tinh chỉnh xác suất ở biên chuẩn.")}'
    })

res_df = pd.DataFrame(out_data)
res_df.to_excel('CreditRating_FeatureAnalysis_v2.xlsx', index=False)
print("Updated CreditRating_FeatureAnalysis_v2.xlsx with professional corporate rating standards.")
