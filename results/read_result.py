import numpy as np
import pandas as pd
import os

def get_latent_dataframe(file_path="results/latent_results.npz"):
    """
    Đọc file .npz và chuyển đổi thành Pandas DataFrame.
    Mỗi hàng ứng với một ngày, chứa 8 giá trị z, 8 mu và 8 var.
    """
    if not os.path.exists(file_path):
        print(f"Lỗi: Không tìm thấy file {file_path}")
        return None

    # 1. Load dữ liệu
    data = np.load(file_path, allow_pickle=True)
    
    # Shapes: (Batch, Time, Z_dim) -> (2745, 20, 8)
    z_all = data['z']
    mu_all = data['mu']
    var_all = data['var']
    dates = data['dates'] if 'dates' in data.files else np.arange(len(z_all))

    # 2. Lấy giá trị tại bước thời gian cuối cùng của mỗi cửa sổ (index = -1)
    # Vì mỗi 'Date' tương ứng với trạng thái cuối cùng của chuỗi T bước
    z_last = z_all[:, -1, :]      # (Batch, 8)
    mu_last = mu_all[:, -1, :]    # (Batch, 8)
    var_last = var_all[:, -1, :]  # (Batch, 8)

    # 3. Tạo danh sách tên cột
    z_cols = [f'z_{i}' for i in range(z_last.shape[1])]
    mu_cols = [f'mu_{i}' for i in range(mu_last.shape[1])]
    var_cols = [f'var_{i}' for i in range(var_last.shape[1])]

    # 4. Tạo DataFrame
    df_z = pd.DataFrame(z_last, columns=z_cols)
    df_mu = pd.DataFrame(mu_last, columns=mu_cols)
    df_var = pd.DataFrame(var_last, columns=var_cols)
    
    df = pd.concat([df_z, df_mu, df_var], axis=1)
    
    # Chèn cột ngày vào đầu
    df.insert(0, 'Date', dates)

    return df

if __name__ == "__main__":
    # Chạy thử nghiệm
    df = get_latent_dataframe()
    
    if df is not None:
        print("\n=== Cấu trúc DataFrame kết quả ===")
        print(df.head())
        print(f"\nKích thước: {df.shape}")
        
        # Lưu ra CSV nếu muốn
        output_csv = "results/latent_analysis_table.csv"
        df.to_csv(output_csv, index=False)
        print(f"\nĐã lưu DataFrame vào: {output_csv}")
        
        # Ví dụ truy vấn: Lấy mu của 5 ngày đầu tiên
        print("\n=== Ví dụ: 5 dòng đầu của các cột 'mu' ===")
        mu_cols = [c for c in df.columns if c.startswith('mu_')]
        print(df[['Date'] + mu_cols].head())
