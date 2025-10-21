from torch import nn


class biLSTM(nn.Module):

    def __init__(self,
                 marker_dim: int = 12,    # Feature Input Size, 4 EE_Pos, 44_EE_Vel, X/Y Pos per Marker, X/Y Vel per Marker
                 hidden_size: int = 256,  # Hidden Layer Size
                 num_markers: int = 9,    # Number of Markers
                 output_dim: int = 2,    # Output Length (X/Y)
                 num_layers: int = 3):    # Number of LSTM Layers
        super(biLSTM, self).__init__()

        self.num_markers = num_markers
        self.marker_dim = marker_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Bidirektional = True
        self.bidirectional = True
        self.num_directions = 2   
        self.fc = nn.Linear(hidden_size * 2, output_dim)     

        # LSTM Layer
        self.lstm = nn.LSTM(
            input_size=marker_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=self.bidirectional
        )


    def forward(self, x):
        """
        x.shape = (batch_size, num_markers, marker_dim)
        """
        # LSTM gibt hidden states für jeden Marker-Schritt zurück
        lstm_out, (h_n, c_n) = self.lstm(x)

        # LSTM Output Shape: (batch_size, num_markers, output_dim)
        output = self.fc(lstm_out)
        
        return output    