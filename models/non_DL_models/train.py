from data import ADM, Particle, Color, load_custom_dataset

# Loading the train and test_loader
train_loader, test_loader, test_x, test_y = ADM(normalize=True, batch_size=1024)    # Loading the ADM dataset
#train_loader, test_loader, test_x, test_y = Color(normalize=True, batch_size=1024)    # Loading the ADM dataset
#train_loader, test_loader, test_x, test_y = Particle(normalize=True, batch_size=1024)    # Loading the ADM dataset

from class_wrapper import SVR, LR, RF

#model = SVR()
#model = LR()
model = RF()
model.train_(train_loader, test_loader)
