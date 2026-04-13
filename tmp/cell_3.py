# Install dependencies for TimeGAN pipeline (Kaggle notebook mode)
%pip install -q --upgrade pip
%pip install -q "tsgm[tensorflow]" scikit-learn pandas numpy scipy matplotlib seaborn

print("Dependencies installed successfully")
print("If needed, restart the kernel and run all cells from the top")