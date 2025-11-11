import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import entropy


class DataPipeline:
    """Pipeline modular: preprocesado → segmentación → características → selección → clasificación"""

    def __init__(self, sampling_rate=200):
        self.sampling_rate = sampling_rate

    # ---------- Paso 1: Preprocesado ----------
    def preprocess(self, data):
        notch_freq = 50.0
        q = 30.0
        b, a = signal.iirnotch(notch_freq, q, self.sampling_rate)
        filtered = data.copy()
        for col in ['Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz']:
            filtered[col] = signal.filtfilt(b, a, data[col])
        return filtered

    # ---------- Paso 2: Segmentación ----------
    def segment(self, data, window_size=50):
        return [data.iloc[i:i+window_size] for i in range(0, len(data), window_size)
                if i+window_size <= len(data)]

    # ---------- Paso 3: Extracción de características ----------
    def extract_features(self, windows):
        feats = []
        for i, w in enumerate(windows):
            f = {
                "Ventana": i + 1,
                "Media_Ax": np.mean(w["Ax"]),
                "RMS_Ax": np.sqrt(np.mean(w["Ax"] ** 2)),
                "Varianza_Ay": np.var(w["Ay"]),
                "Energia_Total": np.sum(w["Ax"] ** 2 + w["Ay"] ** 2 + w["Az"] ** 2),
                "Entropia_Az": self._entropy(w["Az"]),
                "Energia_Rotacional": np.sum(w["Gx"] ** 2 + w["Gy"] ** 2 + w["Gz"] ** 2),
                "Rango_Ax": np.ptp(w["Ax"]),
                "Rango_Gx": np.ptp(w["Gx"])
            }
            feats.append(f)
        return pd.DataFrame(feats)

    # ---------- Paso 4: Selección (simple ejemplo) ----------
    def select_features(self, features):
        # Podrías usar RFE o ANOVA aquí. Por ahora, solo filtramos columnas clave
        selected = features[["Media_Ax", "RMS_Ax", "Varianza_Ay", "Energia_Total", "Entropia_Az"]]
        return selected

    # ---------- Paso 5: Clasificación ----------
    def classify(self, features):
        preds = []
        for _, f in features.iterrows():
            if f["RMS_Ax"] > 0.5 or f["Energia_Total"] > 10:
                preds.append("Saltando")
            else:
                preds.append("Estático")
        return preds

    # ---------- Pipeline completo ----------
    def run_full_pipeline(self, data):
        filtered = self.preprocess(data)
        windows = self.segment(filtered)
        feats = self.extract_features(windows)
        selected = self.select_features(feats)
        preds = self.classify(feats)
        summary = self._build_summary(preds, feats)
        return {"filtered": filtered, "features": feats, "predictions": preds, "summary": summary}

    # ---------- Funciones auxiliares ----------
    def _entropy(self, sig):
        hist, _ = np.histogram(sig, bins=20, density=True)
        hist = hist + 1e-10
        hist /= np.sum(hist)
        return entropy(hist)

    def _build_summary(self, preds, feats):
        total = len(preds)
        counts = pd.Series(preds).value_counts()
        s = f"\nPIPELINE COMPLETADO\n\nTotal de ventanas: {total}\n"
        for c, n in counts.items():
            s += f"{c}: {n} ({n/total*100:.1f}%)\n"
        s += f"\nPromedio RMS_Ax: {feats['RMS_Ax'].mean():.3f}\n"
        s += f"Energía media total: {feats['Energia_Total'].mean():.3f}\n"
        return s
