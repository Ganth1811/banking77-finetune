# Banking Intent Classification using Fine-tuned Llama-3.2-1B

## 1. Project Overview
This project focuses on finetuning a model to classify customer queries within the banking domain. By leveraging the **Llama-3.2-1B-Instruct** model and **Parameter-Efficient Fine-Tuning (PEFT)** techniques, the system effectively maps user utterances to specific banking intents.

## 2. Training Environment (Google Colab)
The training process was conducted on **Google Colab** to utilize cloud-based GPU acceleration.

### Instructions for Training on Colab:
To reproduce the training results, follow these steps:
1.  **Upload Notebook:** Upload the provided training notebook (`train.ipynb`) to your Google Colab environment.
2.  **Upload Datasets:** Ensure that your training and testing datasets (`train.json` and `test.json`) are uploaded into the following directory on Colab:
    - Path: `sample_data/`
3.  **Run Cells:** Execute the cells in order to install dependencies and begin the fine-tuning process.

## 3. Model Configuration
We utilize the **Unsloth** framework to optimize memory usage and training speed.

| Parameter | Value |
| :--- | :--- |
| **Base Model** | `unsloth/Llama-3.2-1B-Instruct` |
| **Max Sequence Length** | 128 |
| **Quantization** | 4-bit (via Unsloth) |

## 4. Training Hyperparameters
| Setting | Value |
| :--- | :--- |
| **Batch Size** | 4 |
| **Gradient Accumulation Steps** | 4 |
| **Learning Rate** | 2e-4 |
| **Optimizer** | `adamw_8bit` |
| **Epochs** | 1 |
| **Seed** | 42 |

### LoRA (Low-Rank Adaptation) Settings
- **Rank (r):** 16
- **Alpha:** 32
- **Target Modules:** `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`

## 5. Inference Guide
After training and saving the model, you can run the inference system locally. We have provided a shell script to simplify the process.

### Running the Inference Script:
1.  **Prepare Environment:** Ensure you have installed the required libraries (contained in `requirements.txt`).
2.  **Grant Execution Permission:** Open your terminal and run the following command to allow the script to execute:
    ```bash
    chmod +x scripts/inference.sh
    ```
3.  **Execute Inference:** Run the script to start the classification interface:
    ```bash
    ./scripts/inference.sh
    ```

The script will trigger the Python inference engine, allowing you to input banking queries and receive predicted intent IDs and labels in real-time.

