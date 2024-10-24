import ast
import csv

def get_best_model(path: str):
    avg_accuracy = {}
    with open(path, mode='r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            key = row[0]
            test_accuracy = ast.literal_eval(row[1])
            val_accuracy = ast.literal_eval(row[2])
            if key in avg_accuracy:
                avg_accuracy[key]  = avg_accuracy[key] + val_accuracy[-1] * 0.2
            else:
                avg_accuracy[key]  = val_accuracy[-1] * 0.2
    best_model = ""
    maximum_avg = 0
    for key, value in avg_accuracy.items():
        if maximum_avg < value:
            best_model = key
            maximum_avg = value
    return best_model, maximum_avg

best_model, maximum_accuracy = get_best_model("./models/results_0.csv")
print(f"best model = {best_model} val_accuracy = {maximum_accuracy}") 
best_model, maximum_accuracy = get_best_model("./models/results_1.csv")
print(f"best model = {best_model} val_accuracy = {maximum_accuracy}") 
best_model, maximum_accuracy = get_best_model("./models/results_2.csv")
print(f"best model = {best_model} val_accuracy = {maximum_accuracy}") 