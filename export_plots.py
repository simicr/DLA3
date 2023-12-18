import matplotlib.pyplot as plt
import pandas as pd
import pickle 
import numpy as np 
import tensorflow as tf

def plot_training_validation_accuracy(model_name):
    df = pd.read_csv(f'results/{model_name}_training_history.csv')

    # Get training and validation accuracy
    train_acc = df['accuracy']
    val_acc = df['val_accuracy']
    epochs = range(1, len(train_acc) + 1)

    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_acc, 'b', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Save the plot as a PNG file
    plt.savefig(f'plots/{model_name}_training_history.png')
    return 

def export_latex_tables(list_csvs):
    # Check if the list of CSVs is empty
    if not list_csvs:
        return None
    
    latex_table = "\\begin{table}[htbp]\n\\centering\n\\begin{tabular}{|c||c|c|c|" + "c|" * (len(pd.read_csv(f'results/{list_csvs[0]}_evaluation_results.csv').columns) - 1) + "}\n\\hline\n"
    
    columns = ["Model"] + pd.read_csv(f'results/{list_csvs[0]}_training_history.csv').columns[:4].tolist()
    latex_table += " & ".join(map(str, columns)) + " \\\\\n\\hline\n"

    for csv_file in list_csvs:
        model_name = csv_file
        data = pd.read_csv(f'results/{csv_file}_training_history.csv')
        data = data.round(3)
        index = data.shape[0]
        print(index)
        for i, row in data.iterrows():
            if i < index - 1: continue
            latex_table += f"{model_name} & " + " & ".join(map(str, row[:4])) + " \\\\\n\\hline\n"

    latex_table += "\\end{tabular}\n\\caption{Your caption here}\n\\end{table}"
    
    return latex_table


dict = pickle.load(open("cifar20_perturb_test.pkl", "rb"))
x_perturb, y_perturb = dict['x_perturb'], dict['y_perturb']
x_perturb = np.mean(x_perturb, axis=3)
x_perturb = np.expand_dims(x_perturb, axis=-1)
y_perturb = tf.keras.utils.to_categorical(y_perturb, num_classes=20)

def explore_augmented_dataset():
    i = 0 
    while i < len(x_perturb):
        fig, axs = plt.subplots(2, 4)
        axs = axs.flatten()
        
        for _, ax in enumerate(axs):
            ax.imshow(x_perturb[i], cmap='gray')
            i+=1
            ax.set_xticks([])
            ax.set_yticks([])

        plt.tight_layout()
        plt.show()

# explore_augmented_dataset()


# models = ['M1', 'M2', 'M3', 'M4', 'M5']
# models = ['M5', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8']
# for m in models:
#     plot_training_validation_accuracy(m)

# models = ['E1-augmented-brigtness', 'E1-augmented-flip', 'E1-augmented-mixed']
models = ['E2']

print(export_latex_tables(models))
