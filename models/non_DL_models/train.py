from data import ADM, Particle, Color, load_custom_dataset
import numpy as np
# Loading the train and test_loader
#train_loader, test_loader, test_x, test_y = ADM(normalize=True, batch_size=1024)    # Loading the ADM dataset
#train_loader, test_loader, test_x, test_y = Color(normalize=True, batch_size=1024)    # Loading the ADM dataset
#train_loader, test_loader, test_x, test_y = Particle(normalize=True, batch_size=1024)    # Loading the ADM dataset

from class_wrapper import SVR, LR, RF, MSE

#model = SVR()
#model = LR()
#model = RF()

model_list = ['LR','SVR','RF']
data_list = ['ADM','Color','Particle']

for model_str in model_list:
    for data in data_list:
        # Load the dataset
        train_loader, test_loader, test_x, test_y = eval('{}(normalize=True, batch_size=1024)'.format(data))    # Loading the ADM dataset
        # Construct the dataset
        model = eval('{}()'.format(model_str))
        # Train the model
        model.train_(train_loader, test_loader)
        # Test the model and output
        pred_y = model(test_x)
        # compare with the testset y
        mse_list = MSE(pred_y, test_y)
        with open('{}_{}.csv'.format(data, model_str), 'w') as f:
            np.savetxt(f, mse_list)
