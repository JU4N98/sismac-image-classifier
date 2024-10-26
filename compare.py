import ast
import csv

def get_best_model(path: str):
    best_model_1 = ""
    maximum_avg = 0

    best_model_2 = ""
    maximum_val = 0
    with open(path, mode='r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            key = row[0]
            test_accuracy = ast.literal_eval(row[1])
            val_accuracy = ast.literal_eval(row[2])
            for i in range(len(test_accuracy)):
                if test_accuracy[i]*0.8 + val_accuracy[i]*0.2 > maximum_avg:
                    maximum_avg = test_accuracy[i]*0.8 + val_accuracy[i]*0.2
                    best_model_1 = key
                if val_accuracy[i] > maximum_val:
                    maximum_val = val_accuracy[i]
                    best_model_2 = key
    
    print(f"Results for {path}")
    print(f"\tbest model by avg accuracy = {best_model_1} avg_accuracy = {maximum_avg}") 
    print(f"\tbest model by val accuracy = {best_model_2} val_accuracy = {maximum_val}") 

print("Model 1:")
get_best_model("./models/model_1/results_0.csv")
get_best_model("./models/model_1/results_1.csv")
get_best_model("./models/model_1/results_2.csv")
print("Model 2:")
get_best_model("./models/model_2/results_0.csv")
get_best_model("./models/model_2/results_1.csv")
get_best_model("./models/model_2/results_2.csv")
