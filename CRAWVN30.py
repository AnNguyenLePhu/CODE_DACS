import pandas as pd
import time
from vnstock import Quote

# ==========================================
# 1. THIẾT LẬP THAM SỐ BAN ĐẦU
# ==========================================
start_date = "2017-01-01"
end_date = "2026-04-19" 

# Danh sách 30 mã cấu thành rổ VN30 hiện tại
vn30_tickers = [
    'ACB', 'BCM', 'BID', 'BVH', 'CTG', 'FPT', 'GAS', 'GVR', 'HDB', 'HPG', 
    'MBB', 'MSN', 'MWG', 'PLX', 'POW', 'SAB', 'SHB', 'SSB', 'SSI', 'STB', 
    'TCB', 'TPB', 'VCB', 'VHM', 'VIB', 'VIC', 'VJC', 'VNM', 'VPB', 'VRE'
]

# ==========================================
# 2. THU THẬP DỮ LIỆU BỐI CẢNH (VN30-INDEX)
# ==========================================
print("Đang tải dữ liệu bối cảnh thị trường (VN30-Index)...")
try:
    vn30_quote = Quote(symbol='VN30', source='VCI')
    vn30_index_df = vn30_quote.history(start=start_date, end=end_date, interval='1D')
    
    vn30_index_df = vn30_index_df.rename(columns={
        'time': 'TradingDate',
        'close': 'VN30_Close',
        'volume': 'VN30_Volume'
    })
    
    vn30_index_df = vn30_index_df[['TradingDate', 'VN30_Close', 'VN30_Volume']]
    print("-> Tải dữ liệu VN30-Index thành công!\n")
except Exception as e:
    print(f"-> Lỗi khi tải VN30-Index: {e}\n")
    vn30_index_df = pd.DataFrame(columns=['TradingDate', 'VN30_Close', 'VN30_Volume'])

# Tránh Rate Limit ngay sau khi tải VN30
time.sleep(3.5)

# ==========================================
# 3. THU THẬP DỮ LIỆU CỔ PHIẾU ĐƠN LẺ (OHLCV)
# ==========================================
print("Bắt đầu tải dữ liệu cho 30 mã cổ phiếu trong rổ VN30...")
all_stocks_data = []

for ticker in vn30_tickers:
    try:
        quote = Quote(symbol=ticker, source='VCI')
        df = quote.history(start=start_date, end=end_date, interval='1D')
        
        if not df.empty:
            df['Ticker'] = ticker
            
            df = df.rename(columns={
                'time': 'TradingDate',
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            })
            
            df = df[['Ticker', 'TradingDate', 'Open', 'High', 'Low', 'Close', 'Volume']]
            all_stocks_data.append(df)
            print(f"  + Tải thành công: {ticker}")
        else:
            print(f"  - Dữ liệu {ticker} rỗng.")
            
    except Exception as e:
        print(f"  x Bỏ qua {ticker} do lỗi: {e}")

    # ÉP VÒNG LẶP NGHỈ 3.5 GIÂY ĐỂ TRÁNH LỖI RATE LIMIT
    time.sleep(3.5)

# ==========================================
# 4. HỢP NHẤT VÀ LƯU TRỮ DỮ LIỆU CUỐI CÙNG
# ==========================================
if len(all_stocks_data) > 0:
    stocks_df = pd.concat(all_stocks_data, ignore_index=True)

    print("\nĐang hợp nhất dữ liệu Cổ phiếu và Chỉ số VN30...")
    final_df = pd.merge(stocks_df, vn30_index_df, on='TradingDate', how='left')

    final_df = final_df[['Ticker', 'TradingDate', 'Open', 'High', 'Low', 'Close', 'Volume', 'VN30_Close', 'VN30_Volume']]

    # Fill dữ liệu trống cho VN30
    final_df[['VN30_Close', 'VN30_Volume']] = final_df[['VN30_Close', 'VN30_Volume']].ffill()

    # Lưu thẳng vào thư mục đồ án DACS của bạn
    output_filename = r'C:\Users\Binh An\OneDrive\Máy tính\CODER\.vscode\Y3\DACS\Dataset_VN30_2017_2026.csv'
    final_df.to_csv(output_filename, index=False)
    
    print(f"\n HOÀN TẤT! Dữ liệu đã được xuất thành công.")
    print(f" Bạn hãy kiểm tra file tại: {output_filename}")
    print(f" Tổng số dòng dữ liệu thu thập được: {len(final_df)} dòng.")
else:
    print("\n[LỖI] Quá trình lấy dữ liệu thất bại, không có mã cổ phiếu nào tải được dữ liệu.")