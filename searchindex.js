Search.setIndex({"docnames": ["Lesson1_FNN", "Lesson2_GPR", "Lesson3_BP", "Lesson4_DeepPot", "Lesson5_BP-FNN_MLP", "Lesson6_DeepPot-FNN_MLP", "Lesson7_BP-GPR_MLP", "index"], "filenames": ["Lesson1_FNN.ipynb", "Lesson2_GPR.ipynb", "Lesson3_BP.ipynb", "Lesson4_DeepPot.ipynb", "Lesson5_BP-FNN_MLP.ipynb", "Lesson6_DeepPot-FNN_MLP.ipynb", "Lesson7_BP-GPR_MLP.ipynb", "index.md"], "titles": ["<span class=\"section-number\">1. </span>Fitting Neural Network Models", "<span class=\"section-number\">2. </span>Gaussian Process Regression Models", "<span class=\"section-number\">3. </span>Behler-Parrinello  Symmetry Functions", "<span class=\"section-number\">4. </span>A PyTorch implementation of Deep Potential-Smooth Edition (DeepPot-SE)", "<span class=\"section-number\">5. </span>Behler-Parrinello Fitting Neural Network with Machine Learning Potential (BP-FNN MLP) Models for the Claisen Rearrangement", "<span class=\"section-number\">6. </span>Lesson 6: DeepPot-Smooth Edition Fitting Neural Network with Machine Learning Potentials (DeepPot-SE-FNN MLP)", "<span class=\"section-number\">7. </span><strong>Lesson 7: Behler-Parrinello Gaussian Process Regression (BP-GPR) for Machine Learning Potentials</strong>", "MLP Tutorial"], "terms": {"In": [0, 1, 2, 5, 6], "thi": [0, 1, 2, 3, 4, 5, 6], "tutori": [0, 1, 2, 4, 5, 6], "we": [0, 1, 2, 3, 4, 5, 6], "learn": [0, 2, 3], "how": [0, 1], "us": [0, 3, 4, 5, 6], "nn": [0, 2, 3, 4, 5, 6], "point": [0, 1], "from": [0, 1, 2, 3], "math": [0, 1, 2, 3, 4, 5, 6], "import": [0, 1], "exp": [0, 1, 2, 4, 6], "pow": [0, 1], "tanh": [0, 1, 2, 3, 4], "numpi": [0, 1, 2, 3, 4, 5, 6], "np": [0, 1, 2, 3, 4, 5, 6], "matplotlib": [0, 1, 2, 3, 4, 5, 6], "pyplot": [0, 1, 2, 3, 4, 5, 6], "plt": [0, 1, 2, 3, 4, 5, 6], "plotli": [0, 1], "graph_object": [0, 1], "go": [0, 1], "For": [0, 1, 4, 5, 6], "definit": [0, 1], "see": [0, 1, 3, 6], "here": [0, 1, 2, 3, 4, 6], "v": [0, 1], "x": [0, 1, 2, 3, 4, 5, 6], "y": [0, 1, 2, 4, 5, 6], "sum_": [0, 1, 2], "k": [0, 1, 2, 3, 4, 5, 6], "0": [0, 1, 2, 3, 4, 5, 6], "a_k": [0, 1], "mathrm": [0, 1], "left": [0, 1, 2, 3], "x_k": [0, 1], "2": [0, 1, 2, 3, 4, 5, 6], "b_k": [0, 1], "y_k": [0, 1], "c_k": [0, 1], "right": [0, 1, 2, 3], "def": [0, 1, 2, 3, 4, 5, 6], "mueller_brown_potenti": [0, 1], "200": [0, 1, 6], "100": [0, 1, 3, 5, 6], "170": [0, 1], "15": [0, 1, 5, 6], "1": [0, 1, 2, 3, 4, 5, 6], "6": [0, 1, 2, 4, 6, 7], "5": [0, 1, 2, 3, 4, 5, 6], "7": [0, 1, 7], "b": [0, 1, 2, 3, 4], "11": [0, 1, 5], "c": [0, 1, 2, 3, 4, 6], "10": [0, 1, 2, 3, 4, 5, 6], "x0": [0, 1], "y0": [0, 1], "valu": [0, 1, 2, 3, 4, 5, 6], "rang": [0, 1, 2, 3, 4, 6], "4": [0, 1, 2, 3, 4, 5, 6], "scale": [0, 1, 3, 6], "make": [0, 1, 4, 5, 6], "easier": [0, 1], "return": [0, 1, 2, 3, 4, 5, 6], "first": [0, 1, 3, 4, 6], "need": [0, 1, 3, 4], "The": [0, 1, 2, 6], "xx": [0, 1], "arang": [0, 1, 2, 4, 6], "8": [0, 1, 2, 4, 5, 6], "yy": [0, 1], "meshgrid": [0, 1, 2, 4, 6], "xy": [0, 1, 2, 4, 5, 6], "xy_trunc": [0, 1], "z": [0, 1], "z_truncat": [0, 1], "now": [0, 1, 2, 3, 4, 5, 6], "arrai": [0, 1, 2, 3, 4, 5, 6], "append": [0, 1, 2, 3, 4, 6], "store": [0, 1, 6], "keep": [0, 1], "onli": [0, 1, 2], "low": [0, 1], "reshap": [0, 1, 2, 3, 4, 5], "len": [0, 1, 2, 3, 4, 5, 6], "so": [0, 2], "can": [0, 1, 2, 3, 4, 5, 6], "our": [0, 1, 2, 3, 4, 5], "print": [0, 1, 2, 4, 5, 6], "zmin": [0, 1], "amin": [0, 1], "zmax": [0, 1], "amax": [0, 1], "size": [0, 2, 3, 6], "test": [0, 1, 6], "set": [0, 2, 3, 5], "futur": [0, 1], "14": [0, 1, 2, 3, 4, 5, 6], "599803525171698": [0, 1], "1194": [0, 1], "4772333054245": [0, 1], "896": [0, 1], "696": [0, 1], "creat": [0, 1], "To": [0, 1, 5], "readabl": [0, 1], "replac": [0, 1], "have": [0, 1, 2, 6], "extrem": [0, 1], "high": [0, 1], "nan": [0, 1], "number": [0, 1, 3, 6], "same": [0, 1], "shape": [0, 1, 2, 3, 4, 6], "help": [0, 1], "ignor": [0, 1, 6], "region": [0, 1], "ar": [0, 1, 2, 3, 4, 5, 6], "interest": [0, 1], "fig": [0, 1, 2, 3, 4, 5, 6], "figur": [0, 1], "colorscal": [0, 1], "rainbow": [0, 1], "cmin": [0, 1], "cmax": [0, 1], "9": [0, 1, 2, 4, 5, 6], "update_trac": [0, 1], "contours_z": [0, 1], "dict": [0, 1], "show": [0, 1, 2, 3], "true": [0, 1, 2, 3, 4, 5, 6], "project_z": [0, 1], "update_layout": [0, 1], "titl": [0, 1, 2], "mueller": 0, "width": [0, 1, 2, 4, 5], "500": [0, 1, 2, 3, 5, 6], "height": [0, 1], "scene": [0, 1], "zaxi": [0, 1], "dtick": [0, 1], "camera_ey": [0, 1], "sinc": 0, "accur": [0, 1, 2], "reflect": [0, 1], "copi": [0, 1], "clean_z": [0, 1], "allow": [0, 1, 2], "an": [0, 1, 2, 3, 6], "figsiz": [0, 1, 2, 4, 5, 6], "dpi": [0, 1, 2, 4, 5, 6], "150": [0, 1], "level": [0, 1, 4, 5, 6], "12": [0, 1, 2, 6], "ct": [0, 1], "color": [0, 1, 2, 3, 4, 5, 6], "clabel": [0, 1], "inlin": [0, 1], "fmt": [0, 1], "0f": [0, 1], "fontsiz": [0, 1, 2], "contourf": [0, 1], "cmap": [0, 1], "cm": [0, 1], "extend": [0, 1, 3, 6], "both": [0, 1], "vmin": [0, 1], "vmax": [0, 1], "xlabel": [0, 1, 2, 4, 5], "labelpad": [0, 1], "75": [0, 1], "ylabel": [0, 1, 2, 4, 5], "tick_param": [0, 1], "axi": [0, 1, 2, 3, 4, 5, 6], "pad": [0, 1], "labels": [0, 1], "cbar": [0, 1], "colorbar": [0, 1], "ax": [0, 1, 2, 3, 4, 5, 6], "m\u00fceller": [0, 1], "tight_layout": [0, 1, 2, 4, 5, 6], "after": [0, 3], "instal": [0, 1, 2, 3, 4], "save": [0, 6], "tensor": [0, 1, 2, 3, 4, 5, 6], "torch": [0, 1, 2, 3, 4, 5, 6], "f": [0, 1, 2, 3, 4, 5, 6], "util": [0, 2, 3, 4, 5, 6], "tensordataset": [0, 2, 3, 4, 5, 6], "dataload": [0, 2, 3, 4, 5, 6], "random_split": [0, 2, 3, 4, 5, 6], "dataset": [0, 2, 3, 4, 5], "train_load": [0, 2, 3, 4, 5], "batch_siz": [0, 2, 3, 4, 5], "32": [0, 1, 2, 3, 4, 5, 6], "shuffl": [0, 1], "below": [0, 1, 3, 4, 5], "schemat": [0, 3], "input": [0, 1, 2, 3, 4, 6], "given": [0, 1, 3], "weight": [0, 2, 3], "w": [0, 2, 3, 4], "each": [0, 1, 2, 3, 4, 5], "neuron": [0, 2, 3, 4, 5], "hidden": 0, "layer": [0, 2, 3, 4], "bia": [0, 2, 3, 4, 5], "ad": 0, "activ": [0, 2, 3, 4], "decid": 0, "bias": 0, "output": [0, 1, 2, 3, 4, 6], "produc": [0, 2, 3], "pred": [0, 6], "python": [0, 1, 5, 6], "_loop": 0, "loop": 0, "through": [0, 2], "neuralnetwork": 0, "modul": [0, 2, 3, 4, 5, 6], "__init__": [0, 1, 2, 3, 4, 5, 6], "self": [0, 1, 2, 3, 4, 5, 6], "n1": [0, 1], "20": [0, 1, 2, 3, 4, 5, 6], "super": [0, 1, 2, 3, 4, 5, 6], "sequenti": [0, 3, 4], "linear": 0, "one": 0, "forward": [0, 1, 2, 3, 4, 5, 6], "train_loop": 0, "optim": [0, 1, 2, 3, 4, 5, 6], "i_epoch": 0, "batch": [0, 2, 3, 4, 5], "enumer": 0, "comput": 0, "loss": [0, 1, 3, 6], "mse_loss": [0, 2, 3, 4, 5], "squeez": [0, 1], "backpropag": [0, 1], "gradient": [0, 2, 3, 4, 5, 6], "updat": [0, 1], "zero_grad": [0, 1, 6], "zero": [0, 1, 6], "out": [0, 1], "previou": [0, 1, 6], "iter": [0, 1, 2, 6], "them": 0, "backward": [0, 1, 6], "step": [0, 1, 2, 4, 5, 6], "current": 0, "item": [0, 1, 6], "epoch": [0, 2, 3, 4, 5], "3d": [0, 1], "3f": [0, 1, 2, 3, 4, 5, 6], "5d": 0, "finish": 0, "when": 0, "desir": [0, 1], "ha": 0, "been": 0, "reach": [0, 1, 5], "also": [0, 1, 2], "some": [0, 2, 4, 5], "term": [0, 2], "pass": 0, "entir": [0, 3], "rate": [0, 1], "determin": 0, "try": 0, "minim": 0, "faster": 0, "would": [0, 2], "larger": [0, 3, 6], "stochast": 0, "descent": 0, "sgd": [0, 1], "algorithm": 0, "learning_r": [0, 2, 3, 4, 5, 6], "1e": [0, 6], "1000": [0, 6], "loss_fn": 0, "lr": [0, 1, 2, 3, 4, 5, 6], "t": [0, 1, 2, 3, 4, 5], "note": [0, 3, 6], "exampl": [0, 2], "broken": 0, "21": [0, 4, 5, 6], "672": 0, "That": 0, "mean": [0, 1, 2, 3, 4, 5, 6], "extra": 0, "24": [0, 6], "togeth": [0, 3], "give": [0, 2, 3], "full": 0, "done": [0, 1, 3], "25": [0, 3, 5, 6], "116": 0, "34": 0, "754": 0, "480": [0, 1], "251": 0, "373": 0, "740": 0, "201": 0, "300": [0, 1, 2, 4, 5, 6], "585": 0, "203": 0, "400": [0, 1, 5], "989": 0, "915": 0, "486": 0, "835": 0, "600": [0, 1], "540": 0, "689": 0, "700": 0, "886": 0, "800": 0, "760": 0, "427": 0, "900": 0, "982": 0, "final": [0, 2, 3, 6], "analyt": [0, 1], "between": [0, 1, 2, 3], "show_surfac": [0, 1], "z_pred": [0, 1], "detach": [0, 1, 2, 3, 4, 5, 6], "zdiff": [0, 1], "subtract": [0, 1], "subplot": [0, 1, 2, 3, 4, 5, 6], "diff": [0, 6], "min": [0, 1, 2], "max": [0, 1, 2, 4, 5, 6], "1182": 0, "7005866913498": 0, "067377484459126": 0, "order": [0, 6], "closer": 0, "plug": 0, "name": [0, 1, 5], "param": [0, 1, 5, 6], "named_paramet": [0, 1], "weights0": 0, "elif": [0, 2, 3, 4], "bias0": 0, "weights2": 0, "bias2": 0, "xy0": 0, "z0": 0, "v1": 0, "i": [0, 1, 2, 3, 4, 5, 6], "v2": 0, "second": 0, "in_featur": [0, 2, 3, 4], "out_featur": [0, 2, 3, 4], "13": [0, 6], "3662": 0, "grad_fn": [0, 6], "addbackward0": 0, "366168556474161": 0, "profession": 0, "infrom": 0, "logs_csv": [0, 2, 3, 4, 5], "directori": 0, "pip": [0, 1, 2, 3, 4, 5, 6], "dev": [0, 2, 3, 4, 5, 6], "null": [0, 2, 3, 4, 5, 6], "pytorch_lightn": [0, 2, 3, 4, 5, 6], "pl": [0, 2, 3, 4, 5, 6], "logger": [0, 2, 3, 4, 5, 6], "pl_logger": [0, 2, 3, 4, 5, 6], "1m": 0, "0m": 0, "49mnotic": 0, "39": 0, "49m": 0, "new": [0, 1], "releas": 0, "avail": [0, 5], "31": [0, 2, 4, 5, 6], "49m23": 0, "run": [0, 1, 6], "49mpip": 0, "upgrad": 0, "you": [0, 1, 6], "mai": 0, "restart": 0, "kernel": [0, 6], "packag": [0, 1, 2, 5, 6], "captur": [0, 1, 2, 3, 4, 5, 6], "pl_model": 0, "lightningmodul": [0, 2, 3, 4, 5], "training_step": [0, 2, 3, 4, 5], "batch_idx": [0, 2, 3, 4, 5], "log": [0, 1, 2, 3, 4, 5, 6], "train_loss": [0, 2, 3, 4, 5], "configure_optim": [0, 2, 3, 4, 5], "adam": [0, 1, 2, 3, 4, 5, 6], "optimz": 0, "instead": [0, 2], "schedul": [0, 2, 3, 4, 5], "adjust": [0, 3], "lr_schedul": [0, 2, 3, 4, 5], "exponentiallr": [0, 2, 3, 4, 5], "95": [0, 2, 3, 4, 5, 6], "interv": [0, 2, 3, 4, 5], "frequenc": [0, 2, 3, 4, 5], "csv_logger": [0, 2, 3, 4, 5], "csvlogger": [0, 2, 3, 4, 5], "trainer": [0, 2, 3, 4, 5], "max_epoch": [0, 2, 3, 4, 5], "gpu": [0, 5], "fals": [0, 1, 2, 3, 4, 5, 6], "tpu": [0, 5], "core": [0, 5], "ipu": [0, 5], "hpu": [0, 5], "type": [0, 2, 3, 4, 5, 6], "81": [0, 6], "trainabl": [0, 5], "non": [0, 5], "total": [0, 3, 4, 5], "000": [0, 1], "estim": [0, 5], "mb": [0, 5], "increas": [0, 1], "With": [0, 1, 4, 5, 6], "should": 0, "decreas": [0, 1], "panda": [0, 2, 3, 4, 5], "pd": [0, 2, 3, 4, 5], "read_csv": [0, 2, 3, 4, 5], "lightning_log": [0, 2, 3, 4, 5], "version_0": [0, 2, 3, 4, 5], "metric": [0, 2, 3, 4, 5], "csv": [0, 2, 3, 4, 5], "semilog": [0, 2, 3, 4, 5], "dodgerblu": [0, 2, 3, 4, 5], "set_xlabel": [0, 2, 3, 4, 5, 6], "set_ylabel": [0, 2, 3, 4, 5, 6], "xtick": [0, 2], "ytick": [0, 2], "again": 0, "1157": 0, "1704667954148": 0, "254873280397252": 0, "start": 1, "function": [1, 3, 5, 7], "deriv": 1, "respect": [1, 3], "muller": 1, "frac": [1, 2, 3], "dv": 1, "dx": 1, "2a_k": 1, "dy": 1, "A": [1, 2], "dx_valu": 1, "dy_valu": 1, "neural": [1, 7], "network": [1, 7], "previous": [1, 2, 3, 4, 5], "lesson": [1, 4, 7], "displai": [1, 4, 5], "dx_truncat": 1, "dy_trunc": 1, "dvx": 1, "dvy": 1, "ml": [1, 4], "x_truncat": 1, "y_truncat": 1, "x_ref": 1, "y_ref": 1, "z_ref": 1, "more": [1, 6], "becaus": [1, 2, 6], "let": 1, "variabl": [1, 6], "vector": [1, 2, 6], "featur": 1, "observ": [1, 2, 6], "textbf": 1, "x_1": 1, "x_d": 1, "n": [1, 3], "configur": 1, "assembl": 1, "_1": [1, 3], "_n": 1, "correspond": 1, "y_1": 1, "y_n": 1, "noisi": 1, "sampl": 1, "assum": 1, "seper": 1, "underli": 1, "accord": 1, "mathit": 1, "\u03b5": 1, "where": [1, 2, 3], "nois": [1, 6], "follow": [1, 2, 4, 5], "distribut": [1, 2, 6], "sim": 1, "mathcal": [1, 3], "\u03c3": 1, "2_n": 1, "sigma": 1, "paramet": [1, 2, 3, 5], "prior": 1, "covarainc": 1, "matrix": [1, 6], "covari": 1, "base": 1, "simular": 1, "begin": [1, 2, 3, 4, 5, 6], "bmatrix": 1, "ldot": 1, "vdot": 1, "ddot": 1, "end": [1, 2, 3], "radial": [1, 2], "basi": 1, "_a": 1, "_b": 1, "2_f": 1, "2l": 1, "vertic": 1, "variat": 1, "l": 1, "length": [1, 2], "setup": 1, "taken": 1, "directli": 1, "minor": 1, "chang": [1, 2], "class": 1, "exactgpmodel": 1, "exactgp": [1, 6], "train_x": 1, "train_i": 1, "likelihood": [1, 6], "mean_modul": [1, 6], "zeromean": 1, "covar_modul": [1, 6], "scalekernel": [1, 6], "rbfkernel": [1, 6], "mean_x": [1, 6], "covar_x": [1, 6], "multivariatenorm": [1, 6], "x_gpr": 1, "z_gpr": 1, "initi": 1, "gaussianlikelihood": [1, 6], "sai": 1, "tild": 1, "ident": 1, "theta": 1, "maxim": 1, "margin": [1, 6], "p": 1, "\u03b8": 1, "pi": [1, 2, 3, 4, 6], "demonstr": 1, "neg": [1, 6], "smooth": [1, 7], "hold": 1, "constant": 1, "vari": 1, "noise_valu": 1, "list": [1, 6], "scale_and_length": 1, "j": [1, 2, 3], "50": [1, 3, 5], "x_plt": 1, "y_plt": 1, "z_plt": 1, "pair": 1, "hyper": 1, "noise_covar": 1, "base_kernel": [1, 6], "lengthscal": [1, 6], "outputscal": [1, 6], "calcul": [1, 2, 4, 5, 6], "mll": [1, 6], "exactmarginalloglikelihood": [1, 6], "call": [1, 2, 4, 5, 6], "tricontour": 1, "tricontourf": 1, "set_label": 1, "rotat": [1, 2, 3, 4], "270": 1, "built": 1, "until": 1, "Then": [1, 6], "train_model": 1, "x_train": [1, 6], "y_train": 1, "print_hp": 1, "untrain": 1, "param_nam": 1, "42": [1, 6], "5f": 1, "training_it": [1, 6], "find": [1, 6], "gp": [1, 6], "raw_nois": 1, "54117": 1, "raw_outputscal": 1, "54132": 1, "raw_lengthscal": 1, "879": 1, "263": 1, "046": 1, "787": 1, "136": 1, "30": [1, 6], "360": [1, 6], "087": 1, "633": 1, "209": [1, 6], "40": [1, 6], "854": 1, "119": 1, "536": 1, "250": 1, "479": 1, "149": 1, "459": 1, "272": 1, "60": [1, 6], "255": 1, "178": 1, "405": 1, "279": 1, "70": 1, "135": 1, "204": 1, "369": 1, "80": [1, 6], "063": 1, "230": 1, "345": 1, "275": 1, "90": 1, "014": 1, "254": 1, "327": 1, "268": 1, "980": 1, "313": 1, "260": 1, "91370": 1, "98492": 1, "03970": 1, "space": [1, 2], "\u03bc": 1, "abov": 1, "equaiton": 1, "eval": [1, 6], "z_var": 1, "47233": 1, "96655": 1, "08929": 1, "30226": 1, "recal": 1, "compos": 1, "depend": [1, 3], "upon": 1, "well": 1, "partial": 1, "themselv": 1, "consequ": 1, "thei": 1, "incorpor": 1, "target": [1, 6], "explicit": 1, "_": [1, 2, 3, 4, 6], "ext": 1, "account": 1, "addit": 1, "displaystyl": 1, "gpmodelwithderiv": 1, "constantmeangrad": 1, "rbfkernelgrad": 1, "ard_num_dim": 1, "multitaskmultivariatenorm": 1, "dx_gpr": 1, "dy_gpr": 1, "y_gpr": 1, "stack": 1, "includ": [1, 6], "x_test": [1, 6], "y_test": 1, "flatten": 1, "multitaskgaussianlikelihood": 1, "num_task": 1, "use_gpu": 1, "cuda": [1, 2, 3, 4, 5, 6], "is_avail": [1, 2, 3, 4, 5, 6], "28": 1, "textrm": [1, 2], "As": 1, "which": [1, 2, 3, 4, 5, 6], "contain": [1, 2, 4], "m": 1, "19": [1, 6], "530": [1, 6], "693": 1, "610": 1, "744": 1, "644": 1, "414": 1, "798": 1, "598": 1, "797": 1, "882": 1, "852": 1, "556": 1, "555": 1, "850": 1, "832": 1, "908": 1, "516": 1, "515": 1, "903": 1, "111": [1, 6], "965": 1, "478": 1, "955": 1, "738": 1, "021": 1, "446": 1, "444": 1, "003": 1, "749": 1, "078": 1, "416": 1, "412": 1, "049": 1, "083": 1, "133": 1, "388": 1, "383": 1, "090": 1, "650": 1, "188": [1, 6], "364": [1, 6], "358": [1, 6], "128": 1, "its": [1, 6], "expect": 1, "e": [1, 2, 3], "2_": 1, "ast_": 1, "ast": 1, "associ": 1, "var": 1, "q": [1, 6], "bigg": 1, "no_grad": 1, "fast_comput": 1, "log_prob": [1, 6], "covar_root_decomposit": 1, "cpu": [1, 2, 3, 4, 5, 6], "n2": 1, "dx_pred": 1, "dx_diff": 1, "dy_pr": 1, "dy_diff": 1, "work": 1, "look": 1, "accuraci": [1, 4, 5, 6], "root": 1, "squar": 1, "error": [1, 2], "rmse": 1, "r": [1, 2, 3, 4, 5, 6], "tabul": 1, "z_test": 1, "evaluate_model": 1, "train_z": 1, "test_x": 1, "test_z": 1, "preds_train": 1, "preds_test": 1, "rmse_train": 1, "sqrt": [1, 2, 3, 4, 5, 6], "rmse_test": 1, "r2": 1, "sum": [1, 2, 3, 4, 5, 6], "q2": [1, 6], "reduce_training_set": 1, "new_siz": 1, "arr_index": 1, "random": [1, 4], "new_model": 1, "size_list": 1, "rmse_train_list": 1, "r2_list": 1, "rmse_test_list": 1, "q2_list": 1, "training_set_s": 1, "set_siz": 1, "training_set_dict": 1, "header": 1, "kei": 1, "floatfmt": 1, "4f": 1, "opt": [1, 5, 6], "hostedtoolcach": [1, 5, 6], "17": [1, 5, 6], "x64": [1, 5, 6], "lib": [1, 5, 6], "python3": [1, 5, 6], "site": [1, 5, 6], "exact_gp": [1, 6], "py": [1, 5, 6], "284": 1, "gpinputwarn": [1, 6], "match": [1, 6], "did": [1, 6], "forget": [1, 6], "0466": 1, "9756": 1, "0532": 1, "9719": 1, "0344": 1, "0620": 1, "9670": 1, "0300": 1, "0721": 1, "9597": 1, "0227": 1, "0888": 1, "9497": 1, "0183": 1, "1125": 1, "9314": 1, "0415": 1, "1158": 1, "9037": 1, "1761": 1, "result": [1, 2], "befor": 1, "rbf": 1, "matern": 1, "ration": 1, "quadrat": [1, 6], "rq": 1, "gpmodel_kernel": 1, "constantmean": [1, 6], "maternkernel": 1, "rqkernel": 1, "train_model_k": 1, "calc": [1, 6], "backprop": [1, 6], "kernel_nam": 1, "str": [1, 2, 3, 4], "split": 1, "0031": 1, "0026": 1, "0021": 1, "atom": [2, 3, 6], "center": 2, "ensur": [2, 3, 4], "invari": [2, 4], "translat": [2, 3, 4], "requir": 2, "lightn": [2, 6], "sequenc": [2, 3, 4, 5, 6], "tupl": [2, 3, 4, 5, 6], "far": 2, "cartesian": 2, "coordin": [2, 3, 6], "machin": [2, 3], "howev": 2, "ideal": 2, "molecul": [2, 3], "case": [2, 3], "chemistri": 2, "impli": 2, "incorrect": 2, "describ": [2, 3, 4], "newli": 2, "transform": [2, 3, 6], "cutoff": [2, 3], "sever": 2, "were": 2, "introduc": 2, "2007": 2, "http": [2, 3, 4, 5, 6], "doi": [2, 3], "org": [2, 3, 6], "1103": 2, "physrevlett": 2, "98": 2, "146401": 2, "align": 2, "f_c": 2, "r_": [2, 3], "ij": 2, "cl": [2, 6], "time": [2, 3, 4, 5], "co": [2, 3, 4, 6], "r_c": [2, 3], "le": 2, "gt": 2, "distanc": [2, 3], "g_i": 2, "neq": 2, "all": [2, 6], "eta": 2, "gaussian": [2, 7], "shift": 2, "peak": 2, "angular": 2, "zeta": [2, 4, 6], "lambda": [2, 6], "theta_": 2, "ijk": [2, 4, 6], "ik": 2, "jk": 2, "3": [2, 3, 4, 5, 6], "mathbf": 2, "cdot": [2, 3], "explor": 2, "environ": [2, 5], "posit": 2, "2017": 2, "isayev": 2, "roitberg": 2, "develop": 2, "anakin": 2, "me": 2, "engin": 2, "molecular": 2, "refer": [2, 3, 4, 5, 6], "1039": 2, "c6sc05720a": 2, "modifi": 2, "shell": 2, "probe": 2, "differ": 2, "angl": [2, 4], "pairwise_vector": [2, 4, 6], "symmetry_function_g2": [2, 4, 6], "bp": [2, 7], "symmetry_function_g3": [2, 4, 6], "symmetry_function_g3ani": [2, 4, 6], "coord": [2, 3, 4, 5, 6], "num_batch": [2, 3, 4, 6], "num_channel": [2, 3, 4, 6], "rij": [2, 3, 4, 6], "none": [2, 3, 4, 5, 6], "mask": [2, 3, 4, 6], "ey": [2, 3, 4, 6], "dtype": [2, 3, 4, 5, 6], "bool": [2, 3, 4, 6], "devic": [2, 3, 4, 5, 6], "remov": [2, 3, 4, 6], "interact": [2, 3, 4, 6], "masked_select": [2, 3, 4, 6], "unsqueez": [2, 3, 4, 6], "view": [2, 3, 4, 6], "rcr": [2, 4, 6], "float": [2, 4, 6], "etar": [2, 4, 6], "shfr": [2, 4, 6], "dij": [2, 3, 4, 6], "norm": [2, 3, 4, 6], "dim": [2, 3, 4, 5, 6], "fij": [2, 4, 6], "g2": [2, 4, 6], "rca": [2, 4, 6], "etaa": [2, 4, 6], "combin": [2, 4, 5, 6], "r12": [2, 4, 6], "r13": [2, 4, 6], "r23": [2, 4, 6], "d12": [2, 4, 6], "d13": [2, 4, 6], "d23": [2, 4, 6], "f12": [2, 4, 6], "f13": [2, 4, 6], "f23": [2, 4, 6], "cosin": [2, 4, 6], "einsum": [2, 4, 6], "ijkl": [2, 4, 6], "g3": [2, 4, 6], "shfz": [2, 4, 6], "shfa": [2, 4, 6], "aco": [2, 4, 6], "feed": 2, "aspect": 2, "connect": [2, 4], "int": [2, 3, 4, 5, 6], "residu": [2, 3, 4, 6], "els": [2, 3, 4, 5, 6], "register_paramet": [2, 3, 4], "reset_paramet": [2, 3, 4], "init": [2, 3, 4], "kaiming_uniform_": [2, 3, 4], "zip": [2, 3, 4, 6], "fan_in": [2, 3, 4], "_calculate_fan_in_and_fan_out": [2, 3, 4], "bound": [2, 3, 4], "uniform_": [2, 3, 4], "channel": [2, 3, 4], "bmm": [2, 3, 4], "transpos": [2, 3, 4, 6], "cat": [2, 3, 4, 6], "rais": [2, 3, 4, 6], "notimplementederror": [2, 3, 4], "Not": [2, 3, 4], "implement": [2, 4, 7], "extra_repr": [2, 3, 4], "format": [2, 3, 4], "take": [2, 4], "inform": [2, 4], "characterist": [2, 4], "uniqu": [2, 3, 4, 5, 6], "assert": [2, 4, 6], "atom_typ": [2, 3, 4, 5, 6], "concat": [2, 4, 6], "properti": [2, 3, 4, 5, 6], "output_length": [2, 3, 4, 5, 6], "similar": [2, 4], "featureani": [2, 4, 6], "n_type": [2, 3, 4], "240": [2, 3, 4, 5], "fitting_net": [2, 3, 4, 5], "train": 2, "descriptor": [2, 3, 4, 5, 6], "5e": [2, 3, 4, 5, 6], "requires_grad_": [2, 3, 4, 5], "atomic_energi": [2, 3, 4, 5], "unbind": [2, 3, 4, 5], "autograd": [2, 3, 4, 5, 6], "grad": [2, 3, 4, 5, 6], "create_graph": [2, 3, 4, 5], "hstack": [2, 3, 4, 5], "qm_coord": [2, 3, 4, 5, 6], "ene_pr": [2, 3, 4, 5], "grad_pr": [2, 3, 4, 5], "ene_loss": [2, 4, 5], "grad_loss": [2, 4, 5], "param_group": [2, 4, 5], "start_lr": [2, 4, 5], "initial_lr": [2, 4, 5], "w_ene": [2, 4, 5], "w_grad": [2, 4, 5], "99": [2, 4, 5], "l2_trn": [2, 4, 5], "l2_e_trn": [2, 4, 5], "l2_f_trn": [2, 4, 5], "validation_step": [2, 4, 5], "set_grad_en": [2, 4, 5, 6], "val_loss": [2, 4, 5], "l2_tst": [2, 4, 5], "l2_e_tst": [2, 4, 5], "l2_f_tst": [2, 4, 5], "github": [2, 3], "npy": [2, 3, 4, 5, 6], "1800": 2, "qm_elem": [2, 4, 5, 6], "txt": [2, 4, 5, 6], "pm3": [2, 4, 5, 6], "energy_sqm": [2, 4, 5, 6], "qm_grad_sqm": [2, 4, 5, 6], "b3lyp": [2, 4, 5, 6], "g": [2, 3, 4, 5, 6], "qm_grad": [2, 4, 5, 6], "These": 2, "provid": 2, "semi": 2, "empir": 2, "qm": 2, "ds": [2, 3, 4, 5, 6], "datasourc": [2, 3, 4, 5, 6], "open": [2, 3, 4, 5, 6], "com": [2, 3, 4, 5, 6], "cc": [2, 3, 4, 5, 6], "ats": [2, 3, 4, 5, 6], "mlp_tutori": [2, 3, 6], "raw": [2, 3, 4, 5, 6], "main": [2, 3, 4, 5, 6], "butan": 2, "rb": [2, 3, 4, 5, 6], "float32": [2, 3, 4, 5, 6], "loadtxt": [2, 3, 4, 5, 6], "elem": [2, 3, 4, 5, 6], "tolist": [2, 3, 4, 5, 6], "index": [2, 3, 4, 5, 6], "repeat": [2, 3, 4, 5, 6], "27": [2, 4, 5, 6], "2114": [2, 4, 5, 6], "23": [2, 4, 5, 6], "061": [2, 4, 5, 6], "qm_gradient": [2, 4, 5, 6], "529177249": [2, 4, 5, 6], "from_numpi": [2, 3, 4, 5, 6], "seed_everyth": [2, 4, 5], "val": [2, 4, 5], "1728": 2, "72": 2, "val_load": [2, 4, 5], "ase_ani": 2, "serv": 2, "interfac": 2, "torchani": [2, 4, 6], "1021": 2, "ac": 2, "jcim": 2, "0c00451": 2, "softwar": 2, "2000e": [2, 4, 6], "00": [2, 4, 6], "5000e": [2, 4, 6], "6000000e": [2, 4, 6], "01": [2, 4, 6], "0000000e": [2, 4, 6], "1687500e": [2, 4, 6], "4375000e": [2, 4, 6], "7062500e": [2, 4, 6], "9750000e": [2, 4, 6], "2437500e": [2, 4, 6], "5125000e": [2, 4, 6], "7812500e": [2, 4, 6], "0500000e": [2, 4, 6], "3187500e": [2, 4, 6], "5875000e": [2, 4, 6], "8562500e": [2, 4, 6], "1250000e": [2, 4, 6], "3937500e": [2, 4, 6], "6625000e": [2, 4, 6], "9312500e": [2, 4, 6], "2000000e": [2, 4, 6], "9634954e": [2, 4, 6], "8904862e": [2, 4, 6], "8174770e": [2, 4, 6], "3744468e": [2, 4, 6], "7671459e": [2, 4, 6], "1598449e": [2, 4, 6], "5525440e": [2, 4, 6], "9452431e": [2, 4, 6], "5500000e": [2, 4, 6], "8500000e": [2, 4, 6], "h": [2, 4, 6], "o": [2, 4, 6], "04": [2, 4, 6], "71": [2, 4, 6], "79": [2, 4, 6], "16": [2, 4, 6], "log_every_n_step": [2, 4, 5], "acceler": [2, 3, 4, 5], "auto": [2, 3, 4, 5], "pt": [2, 4, 5], "model_script": [2, 4, 5], "jit": [2, 4, 5], "attempt": [2, 4, 5], "preserv": [2, 3, 4, 5], "behavior": [2, 4, 5], "oper": [2, 4, 5, 6], "across": [2, 4, 5], "version": [2, 4, 5], "load_state_dict": [2, 4, 5], "state_dict": [2, 4, 5], "to_torchscript": [2, 4, 5], "e1": [2, 4, 5, 6], "e2": [2, 4, 5, 6], "linestyl": [2, 3, 4, 5, 6], "marker": [2, 3, 4, 5, 6], "mediumspringgreen": 2, "concaten": [2, 4, 5, 6], "linewidth": [2, 3, 4, 5, 6], "kcal": [2, 3, 4, 5, 6], "mol": [2, 3, 4, 5, 6], "annot": [2, 4, 5, 6], "05": [2, 4, 5, 6], "xycoord": [2, 4, 5, 6], "fraction": [2, 4, 5, 6], "f1": [2, 3, 4, 5, 6], "f2": [2, 3, 4, 5, 6], "ab": [2, 4, 5, 6], "aa": [2, 3, 4, 5, 6], "savefig": [2, 4, 5, 6], "png": [2, 4, 5, 6], "param_tensor": [2, 4, 5], "33": [2, 6], "48": [2, 4], "isnul": [2, 4, 5], "y2": [2, 4], "valid": 2, "set_titl": 2, "label": [2, 4, 5], "legend": [2, 4, 5], "1350": 2, "xiaoliang": 3, "pan": 3, "2022": 3, "origin": 3, "paper": 3, "symmetri": [3, 7], "inter": 3, "energi": 3, "model": 3, "finit": 3, "system": 3, "linfeng": 3, "zhang": 3, "jiequn": 3, "han": 3, "wang": 3, "wissam": 3, "saidi": 3, "roberto": 3, "car": 3, "weinan": 3, "48550": 3, "arxiv": 3, "1805": 3, "09003": 3, "build": 3, "envrion": 3, "turn": 3, "d_i": 3, "map": 3, "matric": 3, "manner": 3, "permut": [3, 4], "obtain": [3, 4, 5, 6], "extens": 3, "shown": [3, 4], "_i": 3, "eqnarrai": 3, "pmatrix": 3, "s": [3, 4, 5, 6], "1i": 3, "x_": 3, "y_": 3, "z_": 3, "2i": 3, "i1": 3, "ni": 3, "neighbor": 3, "ji": 3, "cs": 3, "down": 3, "becom": 3, "greater": 3, "than": [3, 6], "approach": 3, "It": 3, "beyond": 3, "continu": 3, "differenti": 3, "_environ": 3, "local_environ": 3, "dij_inv": 3, "dij2_inv": 3, "loc_env_r": 3, "loc_env_a": 3, "gener": [3, 4], "compon": 3, "while": 3, "scheme": 3, "present": 3, "m_1": 3, "chemic": 3, "speci": 3, "_2": 3, "two": 3, "dimens": 3, "i2": 3, "m_2": 3, "repres": 3, "By": 3, "multipli": 3, "form": [3, 6], "encod": 3, "d": [3, 6], "yield": 3, "1_i": 3, "2_i": 3, "construct": 3, "axis_neuron": [3, 5], "local_embed": [3, 5], "neighbor_typ": 3, "indic": 3, "load": [3, 4, 5], "deeppot_pytorch": 3, "input_coord": 3, "input_grad": 3, "120": 3, "springgreen": [3, 4, 5], "set_aspect": 3, "equal": 3, "box": 3, "text": 3, "18": [3, 6], "reduct": 3, "bpnn": 4, "delta": [4, 5, 6], "reproduc": [4, 5, 6], "extract": [4, 6], "goal": [4, 5, 6], "semiempir": [4, 5, 6], "dft": [4, 5, 6], "theori": [4, 5, 6], "correct": [4, 5, 6], "what": [4, 5, 6], "d_1": [4, 5], "d_2": [4, 5], "window": [4, 5], "2100": [4, 5, 6], "frame": [4, 5], "ps": [4, 5], "everi": [4, 5], "fs": [4, 5], "ipython": [4, 5], "html": [4, 5, 6], "video": [4, 5], "src": [4, 5], "claisen_rearrang": [4, 5, 6], "img": [4, 5], "mp4": [4, 5], "control": [4, 5], "predict": [4, 5], "2016": [4, 5], "84": [4, 5, 6], "cell": [4, 5, 6], "process": [4, 5, 7], "1152": 4, "365": 4, "0x7f1c0d4e35e0": 4, "claisen": [5, 6, 7], "rearrang": [5, 6, 7], "local": [5, 6], "u": 5, "kora": 5, "wget": 5, "ml_qmmm": 5, "drive": 5, "upload_publ": 5, "url": 5, "modulenotfounderror": 5, "traceback": [5, 6], "most": [5, 6], "recent": [5, 6], "last": [5, 6], "line": [5, 6], "urllib": 5, "request": 5, "urlretriev": 5, "pars": 5, "unquot": 5, "googl": 5, "colab": 5, "mount": 5, "auth_driv": 5, "pydriv": 5, "auth": 5, "googleauth": 5, "No": 5, "propertiess": 5, "info": 5, "lightning_fabr": 5, "seed": 5, "global": 5, "rank_zero": 5, "warn": [5, 6], "csv_log": 5, "miss": 5, "folder": 5, "local_rank": 5, "cuda_visible_devic": 5, "callback": 5, "model_summari": 5, "58": [5, 6], "636": 5, "694": 5, "778": 5, "stop": 5, "model_diff": 5, "model_diff_script": 5, "1575": 5, "content": 5, "mlp": 6, "pytorch": [6, 7], "reaction": 6, "file": 6, "bpgpr": 6, "nskip": 6, "698": 6, "948": 6, "299": 6, "851": 6, "521": 6, "565": 6, "564": 6, "193": 6, "keyboardinterrupt": 6, "__call__": 6, "kwarg": 6, "union": 6, "linearoper": 6, "isinst": 6, "_validate_module_output": 6, "exact_marginal_log_likelihood": 6, "64": 6, "function_dist": 6, "62": 6, "get": 6, "prob": 6, "63": 6, "re": 6, "65": 6, "_add_other_term": 6, "67": 6, "amount": 6, "multivariate_norm": 6, "192": 6, "185": 6, "covar": 6, "186": 6, "diff_siz": 6, "covar_s": 6, "padded_batch_shap": 6, "187": 6, "189": 6, "191": 6, "determinin": 6, "part": 6, "evaluate_kernel": 6, "inv_quad": 6, "logdet": 6, "inv_quad_logdet": 6, "inv_quad_rh": 6, "195": 6, "linear_oper": 6, "added_diag_linear_oper": 6, "addeddiaglinearoper": 6, "208": 6, "added_diag_linear_op": 6, "representation_tre": 6, "represent": 6, "210": 6, "_linear_op": 6, "_diag_tensor": 6, "_linear_oper": 6, "2064": 6, "2054": 6, "linearoperatorrepresentationtre": 6, "2055": 6, "2056": 6, "2057": 6, "obj": 6, "tree": 6, "2062": 6, "subobject": 6, "intern": 6, "2063": 6, "linear_operator_representation_tre": 6, "linear_op": 6, "arg": 6, "itertool": 6, "chain": 6, "_arg": 6, "_differentiable_kwarg": 6, "hasattr": 6, "callabl": 6, "Is": 6, "lazi": 6, "representation_s": 6, "children": 6, "slice": 6, "counter": 6, "lazy_evaluated_kernel_tensor": 6, "397": 6, "lazyevaluatedkerneltensor": 6, "393": 6, "394": 6, "otherwis": 6, "ll": 6, "least": 6, "395": 6, "396": 6, "memoiz": 6, "59": 6, "_cach": 6, "57": 6, "kwargs_pkl": 6, "pickl": 6, "dump": 6, "_is_in_cach": 6, "cache_nam": 6, "_add_to_cach": 6, "method": 6, "_get_from_cach": 6, "recall_grad_st": 6, "wrap": 6, "22": 6, "functool": 6, "_is_grad_en": 6, "26": 6, "355": 6, "353": 6, "temp_active_dim": 6, "active_dim": 6, "354": 6, "356": 6, "x1": 6, "357": 6, "x2": 6, "diag": 6, "359": 6, "last_dim_is_batch": 6, "361": 6, "362": 6, "check": 6, "527": 6, "x1_": 6, "x2_": 6, "528": 6, "529": 6, "to_linear_oper": 6, "531": 6, "532": 6, "scale_kernel": 6, "109": 6, "108": 6, "orig_output": 6, "110": 6, "rbf_kernel": 6, "78": 6, "div": 6, "postprocess_rbf": 6, "covar_dist": 6, "square_dist": 6, "rbfcovari": 6, "appli": 6, "82": 6, "83": 6, "85": 6, "506": 6, "503": 6, "_c": 6, "_are_functorch_transforms_act": 6, "504": 6, "functorch": 6, "vjp": 6, "505": 6, "_functorch": 6, "unwrap_dead_wrapp": 6, "misc": 6, "508": 6, "setup_context": 6, "_singlelevelfunct": 6, "509": 6, "runtimeerror": 6, "510": 6, "511": 6, "vmap": 6, "jvp": 6, "jacrev": 6, "must": 6, "overrid": 6, "512": 6, "staticmethod": 6, "detail": 6, "pleas": 6, "513": 6, "doc": 6, "master": 6, "func": 6, "rbf_covari": 6, "ctx": 6, "sq_dist_func": 6, "unitless_sq_dist": 6, "clone": 6, "inplac": 6, "mess": 6, "unitless_sq_dist_": 6, "needs_grad": 6, "dist_func": 6, "sq_dist": 6, "dist": 6, "x1_eq_x2": 6, "x2_pad": 6, "ones_lik": 6, "x2_norm": 6, "41": 6, "x1_norm": 6, "x1_pad": 6, "43": 6, "matmul": 6, "45": 6, "requires_grad": 6, "prepar": 6, "y_pred": 6, "y_mean": 6, "y_var": 6, "varianc": 6, "y_covar": 6, "covariance_matrix": 6, "ref": 6, "perform": 6, "y_rmse": 6, "y_q2": 6, "usr": 6, "283": 6, "linear_cg": 6, "337": 6, "numericalwarn": 6, "cg": 6, "termin": 6, "averag": 6, "038134537637233734": 6, "toler": 6, "specifi": 6, "cg_toler": 6, "If": 6, "affect": 6, "consid": 6, "maximum": 6, "code": 6, "max_cg_iter": 6, "context": 6, "319": 6, "detect": 6, "like": 6, "due": 6, "numer": 6, "instabl": 6, "round": 6, "up": 6, "06": 6, "1170": 6, "meanbackward0": 6, "1576": 6, "sqrtbackward0": 6, "9982": 6, "rsubbackward1": 6, "auto_grad": 6, "8701": 6, "8277": 6, "qm_coord_train": 6, "fit": 7, "regress": 7, "behler": 7, "parrinello": 7, "deep": 7, "edit": 7, "deeppot": 7, "se": 7, "fnn": 7, "gpr": 7}, "objects": {}, "objtypes": {}, "objnames": {}, "titleterms": {"fit": [0, 2, 3, 4, 5], "neural": [0, 2, 3, 4, 5], "network": [0, 2, 3, 4, 5], "model": [0, 1, 2, 4, 5, 6, 7], "defin": [0, 1, 2, 3, 4, 5, 6], "m\u00fcller": 0, "brown": [0, 1], "potenti": [0, 1, 3, 4, 5, 6, 7], "energi": [0, 1, 2, 4, 5, 6], "function": [0, 2, 4, 6], "gener": [0, 1], "train": [0, 1, 3, 4, 5, 6], "data": [0, 1, 2, 3, 4, 5, 6], "visual": [0, 1], "3": [0, 1], "d": [0, 1], "project": [0, 1], "surfac": [0, 1], "contour": [0, 1], "load": [0, 1, 2, 6], "pytorch": [0, 1, 2, 3, 4, 5], "A": [0, 3, 7], "basic": 0, "class": [0, 2, 3, 4, 5, 6], "plot": [0, 1, 2, 3, 4, 5, 6], "refer": [0, 1], "predict": [0, 1, 2, 3, 6], "differ": [0, 1], "take": 0, "look": 0, "paramet": [0, 4, 6], "more": 0, "autom": 0, "refin": 0, "implement": [0, 3], "lightn": [0, 3, 4, 5], "error": [0, 3, 4], "gaussian": [1, 6], "process": [1, 6], "regress": [1, 6], "mueller": 1, "gpytorch": [1, 6], "gpr": [1, 6], "learn": [1, 4, 5, 6, 7], "hyperparamet": [1, 6], "varianc": 1, "us": [1, 2], "gradient": 1, "perform": 1, "set": [1, 4, 6], "size": [1, 4, 5], "compar": 1, "kernel": 1, "behler": [2, 4, 6], "parrinello": [2, 4, 6], "symmetri": [2, 4, 6], "import": [2, 3, 4, 5, 6], "librari": [2, 3, 4, 5, 6], "featur": [2, 3, 4, 5, 6], "extract": [2, 3, 5], "sequenti": 2, "dens": [2, 3, 4], "creat": [2, 4, 6], "ani": [2, 4, 6], "bpnn": 2, "save": [2, 4, 5], "file": [2, 4, 5], "evalu": [2, 6], "s": 2, "accuraci": 2, "rmsd": [2, 3, 4, 5, 6], "forc": [2, 3, 4, 5, 6], "minim": 2, "loss": [2, 4, 5], "deep": 3, "smooth": [3, 5], "edit": [3, 5], "deeppot": [3, 5], "se": [3, 5], "local": 3, "environ": 3, "embed": 3, "matrix": 3, "The": [3, 4, 5], "machin": [4, 5, 6, 7], "bp": [4, 6], "fnn": [4, 5], "mlp": [4, 5, 7], "claisen": 4, "rearrang": 4, "reaction": [4, 5], "coordin": [4, 5], "system": [4, 5], "from": [4, 5, 6], "github": [4, 5, 6], "mlp_tutori": [4, 5], "seed": 4, "deltamlp": [4, 5], "weight": [4, 5], "bias": [4, 5], "dictionari": [4, 5], "valid": [4, 5], "lesson": [5, 6], "6": 5, "instal": [5, 6], "7": 6, "mlp_class": 6, "reshap": 6, "initi": 6, "rmse": 6, "tabul": 6, "tutori": 7, "tabl": 7, "content": 7, "b": 7, "molecular": 7, "represent": 7, "c": 7}, "envversion": {"sphinx.domains.c": 2, "sphinx.domains.changeset": 1, "sphinx.domains.citation": 1, "sphinx.domains.cpp": 6, "sphinx.domains.index": 1, "sphinx.domains.javascript": 2, "sphinx.domains.math": 2, "sphinx.domains.python": 3, "sphinx.domains.rst": 2, "sphinx.domains.std": 2, "sphinx.ext.intersphinx": 1, "sphinxcontrib.bibtex": 9, "sphinx": 56}})