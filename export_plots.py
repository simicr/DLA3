import matplotlib.pyplot as plt
import pandas as pd

def plot_training_validation_accuracy(model_name):
    df = pd.read_csv(f'{model_name}_training_history.csv')

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
    
    latex_table = "\\begin{table}[htbp]\n\\centering\n\\begin{tabular}{|c||c|" + "c|" * (len(pd.read_csv(f'results/{list_csvs[0]}_evaluation_results.csv').columns) - 1) + "}\n\\hline\n"
    
    columns = ["Model"] + pd.read_csv(f'results/{list_csvs[0]}_evaluation_results.csv').columns[:].tolist()
    latex_table += " & ".join(map(str, columns)) + " \\\\\n\\hline\n"

    for csv_file in list_csvs:
        model_name = csv_file
        data = pd.read_csv(f'results/{csv_file}_evaluation_results.csv')
        for _, row in data.iterrows():
            latex_table += f"{model_name} & " + " & ".join(map(str, row)) + " \\\\\n\\hline\n"

    latex_table += "\\end{tabular}\n\\caption{Your caption here}\n\\end{table}"
    
    return latex_table


# models = ['M5', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8']
models = ['M1', 'M2', 'M3', 'M4', 'M5']
print(export_latex_tables(models))