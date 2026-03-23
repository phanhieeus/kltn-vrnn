import pandas as pd
import numpy as np
import torch


class FeatureBuilder:
    """
    Trình xây dựng đặc trưng (features) cho dữ liệu chứng khoán.
    Chuyển đổi dữ liệu thô (OHLCV) thành các chỉ số kỹ thuật và chuẩn hóa chúng.
    """
    def __init__(self):
        self.features = [
            "log_return",  # Lợi nhuận logarit
            "oc_return",   # Lợi nhuận nội ngày (Open-to-Close)
            "hl_range",    # Biên độ nến (Nến cao/thấp)
            "upper_shadow", # Bóng nến trên
            "lower_shadow", # Bóng nến dưới
            "abs_return",   # Giá trị tuyệt đối của lợi nhuận
            "volatility_10",
            "volatility_20",
            "range_vol",
            "momentum_5",
            "ma_gap",
            "trend_strength",
            "volume",
            "volume_change",
            "volume_z"
        ]

    # -----------------------------
    # BASIC CLEANING
    # -----------------------------
    def parse_price(self, x):
        return float(str(x).replace(",", ""))

    def parse_percent(self, x):
        return float(str(x).replace("%", "")) / 100

    def parse_volume(self, v):
        v = str(v)

        if "M" in v:
            return float(v.replace("M", "")) * 1e6
        elif "K" in v:
            return float(v.replace("K", "")) * 1e3
        else:
            return float(v)

    # -----------------------------
    # LOAD AND CLEAN DATA
    # -----------------------------
    def load_data(self, file_path):
        """
        Đọc dữ liệu từ file CSV, làm sạch tên cột và định dạng lại các giá trị số/ngày tháng.
        Xử lý các ký hiệu như 'M' (triệu), 'K' (nghìn) và dấu phẩy trong giá.
        """
        df = pd.read_csv(file_path)

        df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")

        df["Price"] = df["Price"].apply(self.parse_price)
        df["Open"] = df["Open"].apply(self.parse_price)
        df["High"] = df["High"].apply(self.parse_price)
        df["Low"] = df["Low"].apply(self.parse_price)

        df["Change %"] = df["Change %"].apply(self.parse_percent)

        df["volume"] = df["Vol."].apply(self.parse_volume)

        df = df.sort_values("Date").reset_index(drop=True)

        return df

    # -----------------------------
    # FEATURE ENGINEERING
    # -----------------------------
    def build_features(self, df):
        """
        Tính toán các chỉ số kỹ thuật từ dữ liệu OHLCV.
        Bao gồm: log return, nến, động lượng (momentum), và các chỉ số khối lượng.
        """
        # price dynamics
        df["log_return"] = np.log(df["Price"]).diff()

        df["oc_return"] = (df["Price"] - df["Open"]) / df["Open"]

        df["hl_range"] = (df["High"] - df["Low"]) / df["Price"]

        df["upper_shadow"] = (
            df["High"] - df[["Open", "Price"]].max(axis=1)
        ) / df["Price"]

        df["lower_shadow"] = (
            df[["Open", "Price"]].min(axis=1) - df["Low"]
        ) / df["Price"]

        # volatility
        df["abs_return"] = df["log_return"].abs()

        df["volatility_10"] = df["log_return"].rolling(10).std()

        df["volatility_20"] = df["log_return"].rolling(20).std()

        df["range_vol"] = df["hl_range"].rolling(10).mean()

        # momentum
        df["momentum_5"] = df["Price"].pct_change(5)

        df["ma20"] = df["Price"].rolling(20).mean()

        df["ma_gap"] = (df["Price"] - df["ma20"]) / df["ma20"]

        df["trend_strength"] = (
            df["Price"].rolling(5).mean()
            - df["Price"].rolling(20).mean()
        )

        # volume features
        df["volume_change"] = df["volume"].pct_change()

        df["volume_z"] = (
            df["volume"] - df["volume"].rolling(20).mean()
        ) / df["volume"].rolling(20).std()

        return df

    # -----------------------------
    # NORMALIZATION
    # -----------------------------
    def normalize(self, df):
        """
        Chuẩn hóa (Z-score normalization) các đặc trưng đã chọn.
        Trả về DataFrame đã chuẩn hóa cùng với giá trị trung bình (mean) và độ lệch chuẩn (std).
        """
        # Chỉ chuẩn hóa các cột trong self.features
        data_to_norm = df[self.features].copy()

        mean = data_to_norm.mean()
        std = data_to_norm.std()

        # Thực hiện chuẩn hóa
        feature_df = (data_to_norm - mean) / std
        
        # Giữ lại cột Date nếu có
        if "Date" in df.columns:
            feature_df.insert(0, "Date", df["Date"])

        return feature_df, mean, std

    # -----------------------------
    # FULL PIPELINE
    # -----------------------------
    def build(self, file_path):
        """
        Quy trình xử lý đầy đủ: Load -> Build -> Dropna -> Normalize.
        Dùng để tiền xử lý file CSV thành tập dữ liệu sẵn sàng cho mô hình.
        """
        df = self.load_data(file_path)

        df = self.build_features(df)

        df = df.dropna().reset_index(drop=True)

        feature_df, mean, std = self.normalize(df)

        return df, feature_df, mean, std

    # -----------------------------
    # CONVERT TO TENSOR
    # -----------------------------
    def to_tensor(self, feature_df):

        X = torch.tensor(
            feature_df.values,
            dtype=torch.float32
        )

        X = X.unsqueeze(1)

        return X

    # -----------------------------
    # SAVE FEATURES
    # -----------------------------
    def save_features(self, df, file_path):
        df.to_csv(file_path, index=False)

    # -----------------------------
    # LOAD FEATURES
    # -----------------------------
    def load_features(self, file_path):
        return pd.read_csv(file_path)