# wenyongProj

The target is to predict the IVS for the next time point. We use daily data and get the fixed grid points on surface of each day.
We have 3 methods: 1. pure LSTM(GRU)  2.Conv+LSTM: for each time step, the input go through a convolutional layer, and then go into the lstm, then the output of lstm go through a linear layer and finally reshape to the predicted surface matrix.  3.ConvLSTM: regard the surface matrix as a image and the do the samething.
