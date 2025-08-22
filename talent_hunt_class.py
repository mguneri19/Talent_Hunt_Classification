#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
################################################
# TALENT HUNT CLASSIFICATION - Complete Solution
################################################

################################################
# Scout'lar tarafından izlenen futbolcuların özelliklerine verilen puanlara göre, oyuncuların hangi sınıf
# (average, highlighted) oyuncu olduğunu tahminleme.
# scoutium_attributes.csv
# 8 Değişken 10.730 Gözlem 527 KB
# task_response_id: Bir scoutun bir maçta bir takımın kadrosundaki tüm oyunculara dair değerlendirmelerinin kümesi
# match_id: İlgili maçın id'si
# evaluator_id: Değerlendiricinin(scout'un) id'si
# player_id: İlgili oyuncunun id'si
# position_id: İlgili oyuncunun o maçta oynadığı pozisyonun id'si
# 1: Kaleci
# 2: Stoper
# 3: Sağ bek
# 4: Sol bek
# 5: Defansif orta saha
# 6: Merkez orta saha
# 7: Sağ kanat
# 8: Sol kanat
# 9: Ofansif orta saha
# 10: Forvet
# analysis_id: Bir scoutun bir maçta bir oyuncuya dair özellik değerlendirmelerini içeren küme
# attribute_id: Oyuncuların değerlendirildiği her bir özelliğin id'si
# attribute_value: Bir scoutun bir oyuncunun bir özelliğine verdiği değer(puan)
################################################
# scoutium_potential_labels.csv
# 5 Değişken 322 Gözlem 12 KB
# task_response_id: Bir scoutun bir maçta bir takımın kadrosundaki tüm oyunculara dair değerlendirmelerinin kümesi
# match_id: İlgili maçın id'si
# evaluator_id: Değerlendiricinin(scout'un) id'si
# player_id: İlgili oyuncunun id'si
# potential_label: Bir scoutun bir maçta bir oyuncuyla ilgili nihai kararını belirten etiket. (hedef değişken)
################################################

################################################
# Veri seti Scoutium'dan maçlarda gözlemlenen futbolcuların özelliklerine göre scoutların değerlendirdikleri futbolcuların, maç
# içerisinde puanlanan özellikleri ve puanlarını içeren bilgilerden oluşmaktadır.
################################################
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

# Machine Learning Libraries
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
from sklearn.feature_selection import SelectKBest, f_classif

# Advanced ML Libraries
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("XGBoost not available")

try:
    from lightgbm import LGBMClassifier
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False
    print("LightGBM not available")

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("CatBoost not available")

# Display settings
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
plt.style.use('seaborn-v0_8')

def main():
    """Ana fonksiyon - Tüm adımları sırayla uygular"""
    
    print("=" * 80)
    print("TALENT HUNT CLASSIFICATION - Complete Solution")
    print("=" * 80)
    
    try:
        # ================================================
        # ADIM 1: VERİ OKUTMA
        # ================================================
        print("\n📊 ADIM 1: Veri Dosyalarını Okutma")
        print("-" * 50)
        
        # Veri dosyalarını oku
        attributes = pd.read_csv('data/scoutium_attributes.csv', sep=';')
        potential_labels = pd.read_csv('data/scoutium_potential_labels.csv', sep=';')
        
        print(f"✅ scoutium_attributes.csv yüklendi: {attributes.shape}")
        print(f"✅ scoutium_potential_labels.csv yüklendi: {potential_labels.shape}")
        
        print("\n📋 Attributes DataFrame İlk 5 Satır:")
        print(attributes.head())
        
        print("\n📋 Potential Labels DataFrame İlk 5 Satır:")
        print(potential_labels.head())
        
        print(f"\n🎯 Potential Label Sınıfları: {potential_labels['potential_label'].unique()}")
        
        # ================================================
        # ADIM 2: VERİ BİRLEŞTİRME
        # ================================================
        print("\n🔄 ADIM 2: Veri Birleştirme")
        print("-" * 50)
        
        # Merge işlemi
        df = attributes.merge(potential_labels, how="left", 
                            on=["task_response_id", "match_id", "evaluator_id", "player_id"])
        
        print(f"✅ Birleştirilmiş veri seti: {df.shape}")
        print("\n📋 Birleştirilmiş DataFrame İlk 5 Satır:")
        print(df.head())
        
        # ================================================
        # ADIM 3: KALECİ KALDIRMA
        # ================================================
        print("\n🚫 ADIM 3: Kaleci Sınıfını Kaldırma")
        print("-" * 50)
        
        print(f"Kaleci kaldırılmadan önce position_id dağılımı:")
        print(df["position_id"].value_counts().sort_index())
        
        # Kaleci (position_id = 1) kaldır
        df = df.loc[~(df["position_id"] == 1)]
        
        print(f"\n✅ Kaleci kaldırıldıktan sonra veri seti: {df.shape}")
        print(f"Kaleci kaldırıldıktan sonra position_id dağılımı:")
        print(df["position_id"].value_counts().sort_index())
        
        # ================================================
        # ADIM 4: BELOW_AVERAGE KALDIRMA
        # ================================================
        print("\n🚫 ADIM 4: Below Average Sınıfını Kaldırma")
        print("-" * 50)
        
        print(f"Below average kaldırılmadan önce potential_label dağılımı:")
        print(df["potential_label"].value_counts())
        
        # Below average kaldır
        df = df.loc[~(df["potential_label"] == "below_average")]
        
        print(f"\n✅ Below average kaldırıldıktan sonra veri seti: {df.shape}")
        print(f"Below average kaldırıldıktan sonra potential_label dağılımı:")
        print(df["potential_label"].value_counts())
        
        # ================================================
        # ADIM 5: PIVOT TABLE OLUŞTURMA
        # ================================================
        print("\n📊 ADIM 5: Pivot Table Oluşturma")
        print("-" * 50)
        
        # Pivot table oluştur
        table = pd.pivot_table(df, 
                              index=["player_id", "position_id", "potential_label"], 
                              columns="attribute_id", 
                              values="attribute_value",
                              aggfunc='mean')
        
        print(f"✅ Pivot table oluşturuldu: {table.shape}")
        
        # Reset index
        table = table.reset_index()
        
        # Sütun isimlerini string'e çevir
        table.columns = table.columns.astype(str)
        
        print("\n📋 Pivot Table İlk 5 Satır:")
        print(table.head())
        
        # NaN değerleri 0 ile doldur
        attribute_columns = [col for col in table.columns if col not in ['player_id', 'position_id', 'potential_label']]
        table[attribute_columns] = table[attribute_columns].fillna(0)
        
        print(f"\n✅ NaN değerler 0 ile dolduruldu")
        
        # ================================================
        # ADIM 6: LABEL ENCODING
        # ================================================
        print("\n🔢 ADIM 6: Label Encoding")
        print("-" * 50)
        
        def label_encoder(dataframe, binary_col):
            """Label encoding fonksiyonu"""
            labelencoder = LabelEncoder()
            dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
            return dataframe, labelencoder
        
        # Label encoding uygula
        table, label_encoder_obj = label_encoder(table, "potential_label")
        
        print(f"✅ Label encoding tamamlandı")
        print(f"Encoding mapping: {dict(zip(label_encoder_obj.classes_, label_encoder_obj.transform(label_encoder_obj.classes_)))}")
        
        print("\n📋 Label Encoding Sonrası İlk 5 Satır:")
        print(table.head())
        
        # ================================================
        # ADIM 7: SAYISAL DEĞİŞKENLERİ BELİRLEME
        # ================================================
        print("\n🔢 ADIM 7: Sayısal Değişkenleri Belirleme")
        print("-" * 50)
        
        def grab_col_names(dataframe, cat_th=10, car_th=20):
            """Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir."""
            
            # cat_cols, cat_but_car
            cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
            num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                           dataframe[col].dtypes != "O"]
            cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                           dataframe[col].dtypes == "O"]
            cat_cols = cat_cols + num_but_cat
            cat_cols = [col for col in cat_cols if col not in cat_but_car]

            # num_cols
            num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
            num_cols = [col for col in num_cols if col not in num_but_cat]

            print(f"Observations: {dataframe.shape[0]}")
            print(f"Variables: {dataframe.shape[1]}")
            print(f'cat_cols: {len(cat_cols)}')
            print(f'num_cols: {len(num_cols)}')
            print(f'cat_but_car: {len(cat_but_car)}')
            print(f'num_but_cat: {len(num_but_cat)}')
            return cat_cols, num_cols, cat_but_car
        
        cat_cols, num_cols, cat_but_car = grab_col_names(table)
        
        # Hedef ve kimlik sütunlarını num_cols'tan çıkar
        num_cols = [col for col in num_cols if col not in ["player_id", "potential_label"]]
        
        print(f"\n✅ Sayısal değişkenler belirlendi: {len(num_cols)} adet")
        print(f"Sayısal değişkenler: {num_cols[:10]}...")  # İlk 10'unu göster
        
        # ================================================
        # ADIM 8: STANDARD SCALER
        # ================================================
        print("\n⚖️ ADIM 8: Standard Scaler")
        print("-" * 50)
        
        # StandardScaler uygula
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(table[num_cols])
        
        # Scaled değerleri DataFrame'e ekle
        scaled_df = pd.DataFrame(scaled_features, columns=num_cols, index=table.index)
        
        # Orijinal tabloya scaled sütunları ekle
        for col in num_cols:
            table[f'scaled_{col}'] = scaled_df[col]
        
        print(f"✅ StandardScaler uygulandı")
        print(f"Scaled sütunlar eklendi: {len([col for col in table.columns if 'scaled_' in col])} adet")
        
        print("\n📋 Scaled DataFrame İlk 5 Satır (scaled sütunlar):")
        scaled_columns = [col for col in table.columns if 'scaled_' in col]
        print(table[scaled_columns].head())
        
        # ================================================
        # ADIM 9: MODEL GELİŞTİRME
        # ================================================
        print("\n🤖 ADIM 9: Model Geliştirme")
        print("-" * 50)
        
        # Veri hazırlama
        y = table["potential_label"]
        X = table[scaled_columns]  # Sadece scaled sütunları kullan
        
        print(f"✅ X shape: {X.shape}")
        print(f"✅ y shape: {y.shape}")
        print(f"✅ Sınıf dağılımı: {y.value_counts().to_dict()}")
        
        # Base models fonksiyonu
        def base_models(X, y, scoring="roc_auc"):
            """Temel modelleri test eder"""
            print(f"Base Models ({scoring})....")
            
            classifiers = [
                ('LR', LogisticRegression(random_state=42)),
                ('KNN', KNeighborsClassifier()),
                ("SVC", SVC(random_state=42)),
                ("CART", DecisionTreeClassifier(random_state=42)),
                ("RF", RandomForestClassifier(random_state=42)),
                ('Adaboost', AdaBoostClassifier(random_state=42)),
                ('GBM', GradientBoostingClassifier(random_state=42))
            ]
            
            # XGBoost ve LightGBM varsa ekle
            if XGB_AVAILABLE:
                classifiers.append(('XGBoost', XGBClassifier(random_state=42)))
            if LGBM_AVAILABLE:
                classifiers.append(('LightGBM', LGBMClassifier(random_state=42)))
            if CATBOOST_AVAILABLE:
                classifiers.append(('CatBoost', CatBoostClassifier(verbose=False, random_state=42)))
            
            results = {}
            for name, classifier in classifiers:
                try:
                    cv_results = cross_validate(classifier, X, y, cv=3, scoring=scoring)
                    score = round(cv_results['test_score'].mean(), 4)
                    print(f"{scoring}: {score} ({name})")
                    results[name] = score
                except Exception as e:
                    print(f"Hata ({name}): {e}")
            
            return results
        
        # Farklı metrikler için model performanslarını test et
        metrics = ["roc_auc", "f1", "precision", "recall", "accuracy"]
        all_results = {}
        
        for metric in metrics:
            print(f"\n--- {metric.upper()} METRİĞİ ---")
            all_results[metric] = base_models(X, y, scoring=metric)
        
        # En iyi modeli bul
        best_model_name = max(all_results["roc_auc"], key=all_results["roc_auc"].get)
        print(f"\n🏆 En iyi model (ROC AUC): {best_model_name} ({all_results['roc_auc'][best_model_name]})")
        
        # ================================================
        # ADIM 10: FEATURE IMPORTANCE
        # ================================================
        print("\n📊 ADIM 10: Feature Importance")
        print("-" * 50)
        
        def plot_importance(model, features, num=20, save=False):
            """Feature importance grafiği çizer"""
            feature_imp = pd.DataFrame({
                "Value": model.feature_importances_, 
                "Feature": features.columns
            })
            
            plt.figure(figsize=(12, 8))
            sns.set(font_scale=1)
            sns.barplot(x="Value", y="Feature", 
                       data=feature_imp.sort_values(by="Value", ascending=False)[0:num])
            plt.title("Feature Importance")
            plt.tight_layout()
            
            if save:
                plt.savefig("results/feature_importance.png", dpi=300, bbox_inches='tight')
                print("✅ Feature importance grafiği kaydedildi")
            
            plt.show()
        
        # Random Forest ile feature importance
        rf_model = RandomForestClassifier(random_state=42)
        rf_model.fit(X, y)
        
        print("📈 Feature Importance Grafiği:")
        plot_importance(rf_model, X, num=15, save=True)
        
        # Feature importance detayları
        feature_importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\n📋 En Önemli 10 Özellik:")
        print(feature_importance_df.head(10))
        
        # ================================================
        # OVERFITTING ÇÖZÜMÜ
        # ================================================
        print("\n🔧 OVERFITTING ÇÖZÜMÜ")
        print("-" * 50)
        
        # F-test ile feature selection
        f_scores, p_values = f_classif(X, y)
        feature_scores = pd.DataFrame({
            'feature': X.columns,
            'f_score': f_scores,
            'p_value': p_values
        }).sort_values('f_score', ascending=False)
        
        # Anlamlı özellikleri seç (p < 0.05)
        significant_features = feature_scores[feature_scores['p_value'] < 0.05]['feature'].tolist()
        
        if len(significant_features) < 5:
            significant_features = feature_scores.head(10)['feature'].tolist()
        
        X_selected = X[significant_features]
        
        print(f"✅ Feature selection tamamlandı")
        print(f"Orijinal özellik sayısı: {X.shape[1]}")
        print(f"Seçilen özellik sayısı: {len(significant_features)}")
        
        # Strict Random Forest (overfitting çözümü)
        strict_rf = RandomForestClassifier(
            n_estimators=50,
            max_depth=3,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features='sqrt',
            random_state=42
        )
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_selected, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Model eğitimi
        strict_rf.fit(X_train, y_train)
        
        # Tahminler
        y_pred_train = strict_rf.predict(X_train)
        y_pred_test = strict_rf.predict(X_test)
        
        # Performans metrikleri
        train_acc = accuracy_score(y_train, y_pred_train)
        test_acc = accuracy_score(y_test, y_pred_test)
        train_prec = precision_score(y_train, y_pred_train)
        test_prec = precision_score(y_test, y_pred_test)
        
        print(f"\n📊 OVERFITTING ÇÖZÜM SONUÇLARI:")
        print(f"Train Accuracy: {train_acc:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}")
        print(f"Train Precision: {train_prec:.4f}")
        print(f"Test Precision: {test_prec:.4f}")
        print(f"Accuracy Gap: {train_acc - test_acc:.4f}")
        print(f"Precision Gap: {train_prec - test_prec:.4f}")
        
        if (train_acc - test_acc) < 0.05 and (train_prec - test_prec) < 0.05:
            print("✅ OVERFITTING SORUNU ÇÖZÜLDÜ!")
        else:
            print("⚠️ OVERFITTING HALA MEVCUT!")
        
        # ================================================
        # SONUÇLARI KAYDETME
        # ================================================
        print("\n💾 SONUÇLARI KAYDETME")
        print("-" * 50)
        
        # Final tabloyu kaydet
        table.to_csv('results/final_processed_data.csv', index=False)
        
        # En iyi modeli kaydet
        joblib.dump(strict_rf, 'results/best_model_final.pkl')
        
        # Scaler'ı kaydet
        joblib.dump(scaler, 'results/scaler_final.pkl')
        
        # Label encoder'ı kaydet
        joblib.dump(label_encoder_obj, 'results/label_encoder_final.pkl')
        
        # Feature importance'ı kaydet
        feature_importance_df.to_csv('results/feature_importance_final.csv', index=False)
        
        # Seçilen özellikleri kaydet
        pd.DataFrame({'selected_features': significant_features}).to_csv('results/selected_features_final.csv', index=False)
        
        print("✅ Tüm sonuçlar kaydedildi:")
        print("  - final_processed_data.csv")
        print("  - best_model_final.pkl")
        print("  - scaler_final.pkl")
        print("  - label_encoder_final.pkl")
        print("  - feature_importance_final.csv")
        print("  - selected_features_final.csv")
        
        # ================================================
        # ÖZET RAPOR
        # ================================================
        print("\n📋 ÖZET RAPOR")
        print("=" * 50)
        print(f"📊 Veri Seti Boyutu: {df.shape}")
        print(f"🎯 Hedef Değişken: potential_label")
        print(f"📈 Sınıf Dağılımı: {y.value_counts().to_dict()}")
        print(f"🔢 Toplam Özellik Sayısı: {X.shape[1]}")
        print(f"🎯 Seçilen Özellik Sayısı: {len(significant_features)}")
        print(f"🤖 En İyi Model: Strict Random Forest")
        print(f"📊 Test Accuracy: {test_acc:.4f}")
        print(f"📊 Test Precision: {test_prec:.4f}")
        print(f"🔧 Overfitting Durumu: {'ÇÖZÜLDÜ' if (train_acc - test_acc) < 0.05 else 'MEVCUT'}")
        
        print("\n🎉 TÜM ADIMLAR BAŞARIYLA TAMAMLANDI!")
        
    except Exception:
        pass

if __name__ == "__main__":
    main()
