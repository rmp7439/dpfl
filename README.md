# DP-FL

**Goal:** DP-FL: Non-IID Dirichlet split + Opacus DP-SGD + Flower federation + RDP accounting.

## Setup Instructions (Google Colab)

1. Open a new Google Colab notebook and set the hardware accelerator to **GPU** (Runtime > Change runtime type > T4 GPU).
2. Clone this repository into the Colab environment:
   ```bash
   !git clone https://github.com/rmp7439/dpfl.git
   %cd dpfl
   ```
3. Install the pinned dependencies:
   ```bash
   !pip install -r requirements.txt
   ```
4. Run the initial GPU and setup validation:
   ```bash
   !python main.py
   ```
