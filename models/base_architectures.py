import torch

class MipNeRFModel(torch.nn.Module):
    def __init__(
        self,
        num_layers=8,
        hidden_size=256,
        skip_connect_every=4,
        max_ipe_deg=16,
        num_encoding_fn_dir=4,
        include_input_xyz=False,
        include_input_dir=False,
        use_viewdirs=True,
    ):
        super(MipNeRFModel, self).__init__()

        include_input_xyz = 3 if include_input_xyz else 0
        include_input_dir = 3 if include_input_dir else 0
        self.dim_xyz = include_input_xyz + 2 * 3 * max_ipe_deg
        self.dim_dir = include_input_dir + 2 * 3 * num_encoding_fn_dir
        self.number_of_viewdir_layers = 1
        self.layers_xyz = torch.nn.ModuleList()
        self.use_viewdirs = use_viewdirs
        self.layers_xyz.append(torch.nn.Linear(self.dim_xyz, hidden_size))
        for i in range(1,8):
            if i == 5:
                self.layers_xyz.append(torch.nn.Linear(self.dim_xyz + hidden_size, hidden_size))
            else:
                self.layers_xyz.append(torch.nn.Linear(hidden_size, hidden_size))
        self.fc_feat = torch.nn.Linear(hidden_size, hidden_size)
        self.fc_alpha = torch.nn.Linear(hidden_size, 1)

        self.layers_dir = torch.nn.ModuleList()
        self.layers_dir.append(torch.nn.Linear(hidden_size + self.dim_dir, 128))
        for i in range(self.number_of_viewdir_layers-1):
            self.layers_dir.append(torch.nn.Linear(128, 128))
        self.fc_rgb = torch.nn.Linear(128, 3)
        self.relu = torch.nn.functional.relu

    def forward(self, x):
        xyz, dirs = x[..., : self.dim_xyz], x[..., self.dim_xyz :]
        x = self.layers_xyz[0](xyz)
        x = self.relu(x)
        for i in range(1, 8):
            if i == 5:
                x = self.layers_xyz[i](torch.cat((xyz, x), -1))
            else:
                x = self.layers_xyz[i](x)
            x = self.relu(x)
        feat = self.fc_feat(x)
        alpha = self.fc_alpha(feat)
        if self.use_viewdirs:
            x = self.layers_dir[0](torch.cat((feat, dirs), -1))
        else:
            x = self.layers_dir[0](feat)
        x = self.relu(x)
        for i in range(self.number_of_viewdir_layers-1):
            x = self.layers_dir[i](x)
            x = self.relu(x)
        rgb = self.fc_rgb(x)
        return torch.cat((rgb, alpha), dim=-1)


class DepthMipNeRFModel(torch.nn.Module):
    def __init__(
        self,
        num_layers=8,
        hidden_size=256,
        skip_connect_every=4,
        max_ipe_deg=16,
        num_encoding_fn_dir=4,
        include_input_xyz=False,
        include_input_dir=False,
        use_viewdirs=True,
    ):
        super(DepthMipNeRFModel, self).__init__()

        include_input_xyz = 3 if include_input_xyz else 0
        include_input_dir = 3 if include_input_dir else 0
        self.dim_xyz = include_input_xyz + 2 * 3 * max_ipe_deg
        self.dim_dir = include_input_dir + 2 * 3 * num_encoding_fn_dir
        self.number_of_viewdir_layers = 1
        self.layers_xyz = torch.nn.ModuleList()
        self.use_viewdirs = use_viewdirs
        self.layers_xyz.append(torch.nn.Linear(self.dim_xyz, hidden_size))
        for i in range(1, 8):
            if i == 5:
                self.layers_xyz.append(torch.nn.Linear(self.dim_xyz + hidden_size, hidden_size))
            else:
                self.layers_xyz.append(torch.nn.Linear(hidden_size, hidden_size))
        self.fc_feat = torch.nn.Linear(hidden_size, hidden_size)
        self.fc_alpha = torch.nn.Linear(hidden_size, 1)

        self.layers_dir = torch.nn.ModuleList()
        self.layers_dir.append(torch.nn.Linear(hidden_size + self.dim_dir, 128))
        for i in range(self.number_of_viewdir_layers-1):
            self.layers_dir.append(torch.nn.Linear(128, 128))
        self.fc_rgb = torch.nn.Linear(128, 3)
        self.fc_mu_sigma = torch.nn.Linear(128, 2)
        #self.new_fc_mu_sigma = torch.nn.Linear(256, 2)
        self.relu = torch.nn.functional.relu

    def forward(self, x):
        xyz, dirs = x[..., : self.dim_xyz], x[..., self.dim_xyz:]
        x = self.layers_xyz[0](xyz)
        x = self.relu(x)
        for i in range(1, 8):
            if i == 5:
                x = self.layers_xyz[i](torch.cat((xyz, x), -1))
            else:
                x = self.layers_xyz[i](x)
            x = self.relu(x)
        feat = self.fc_feat(x)
        alpha = self.fc_alpha(feat)
        if self.use_viewdirs:
            x = self.layers_dir[0](torch.cat((feat, dirs), -1))
        else:
            x = self.layers_dir[0](feat)
        x = self.relu(x)
        for i in range(self.number_of_viewdir_layers-1):
            x = self.layers_dir[i](x)
            x = self.relu(x)
        rgb = self.fc_rgb(x)
        mu_sigma = self.fc_mu_sigma(x)
        #mu_sigma = self.new_fc_mu_sigma(feat)
        return torch.cat((rgb, alpha, mu_sigma), dim=-1)