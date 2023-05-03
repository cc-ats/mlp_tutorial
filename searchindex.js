Search.setIndex({"docnames": ["Lesson1_FNN", "Lesson2_GPR", "Lesson3_BP", "Lesson4_DeepPot", "Lesson5_BP-FNN_MLP", "Lesson6_DeepPot-FNN_MLP", "Lesson7_BP-GPR_MLP", "index"], "filenames": ["Lesson1_FNN.ipynb", "Lesson2_GPR.ipynb", "Lesson3_BP.ipynb", "Lesson4_DeepPot.ipynb", "Lesson5_BP-FNN_MLP.ipynb", "Lesson6_DeepPot-FNN_MLP.ipynb", "Lesson7_BP-GPR_MLP.ipynb", "index.md"], "titles": ["<span class=\"section-number\">1. </span>Fitting Neural Network Models", "<span class=\"section-number\">2. </span>Gaussian Process Regression Models", "<span class=\"section-number\">3. </span>Behler-Parrinello  Symmetry Functions", "<span class=\"section-number\">4. </span>A PyTorch implementation of Deep Potential-Smooth Edition (DeepPot-SE)", "<span class=\"section-number\">5. </span>Neural Netwotk with Behler-Parrinello Symmetry Functions", "<span class=\"section-number\">6. </span>Neural Netwotk with Deep Potential", "<span class=\"section-number\">7. </span>Gaussian Process Regression with Behler-Parrinello Symmetry Functions", "MLP Class"], "terms": {"In": [0, 1, 2, 3, 5], "thi": [0, 1, 2, 3], "tutori": [0, 1, 2], "we": [0, 1, 2, 3], "learn": [0, 2], "how": [0, 1], "us": [0, 3, 6], "point": [0, 1], "from": [0, 1, 3, 4, 5, 6], "math": [0, 1, 2, 3, 4, 5, 6], "import": [0, 1, 4, 5, 6], "exp": [0, 1, 2, 4, 6], "pow": [0, 1], "tanh": [0, 1, 2, 3, 4], "numpi": [0, 1, 2, 3, 4, 5, 6], "np": [0, 1, 2, 3, 4, 5, 6], "matplotlib": [0, 1, 2, 3, 4, 5, 6], "pyplot": [0, 1, 2, 3, 4, 5, 6], "plt": [0, 1, 2, 3, 4, 5, 6], "plotli": [0, 1], "graph_object": [0, 1], "go": [0, 1], "For": [0, 1], "definit": [0, 1], "see": [0, 1, 3], "here": [0, 1, 2, 3], "v": [0, 1], "x": [0, 1, 2, 3, 4, 5, 6], "y": [0, 1, 2, 4, 5, 6], "sum_": [0, 1, 2], "k": [0, 1, 2, 3, 4, 5, 6], "0": [0, 1, 2, 3, 4, 5, 6], "a_k": [0, 1], "mathrm": [0, 1], "left": [0, 1, 2, 3], "x_k": [0, 1], "2": [0, 1, 2, 3, 4, 5, 6], "b_k": [0, 1], "y_k": [0, 1], "c_k": [0, 1], "right": [0, 1, 2, 3], "def": [0, 1, 2, 3, 4, 5, 6], "mueller_brown_potenti": [0, 1], "200": [0, 1], "100": [0, 1, 3, 5], "170": [0, 1], "15": [0, 1], "1": [0, 1, 2, 3, 4, 5, 6], "6": [0, 1, 2, 4, 5, 6], "5": [0, 1, 2, 3, 4, 5, 6], "7": [0, 1, 3], "b": [0, 1, 2, 3, 4], "11": [0, 1, 3, 5], "c": [0, 1, 2, 3, 4, 6], "10": [0, 1, 2, 3, 4, 5, 6], "x0": [0, 1], "y0": [0, 1], "valu": [0, 1, 2, 3], "rang": [0, 1, 2, 3, 4, 6], "4": [0, 1, 2, 3, 4, 5, 6], "scale": [0, 1, 3], "make": [0, 1, 6], "easier": 0, "return": [0, 1, 2, 3, 4, 5, 6], "first": [0, 1, 2, 3], "need": [0, 1, 2, 3], "xx": [0, 1], "arang": [0, 1, 2, 4, 6], "8": [0, 1, 2, 4, 5, 6], "yy": [0, 1], "meshgrid": [0, 1, 2, 4, 6], "xy": [0, 1, 2, 4, 5, 6], "xy_trunc": [0, 1], "z": [0, 1], "z_truncat": [0, 1], "now": [0, 1, 2, 3], "arrai": [0, 1, 2, 3, 4, 5, 6], "append": [0, 1, 2, 3, 4], "store": [0, 1], "keep": [0, 1], "onli": [0, 1, 2], "low": [0, 1], "reshap": [0, 1, 2, 3, 4, 5, 6], "len": [0, 1, 2, 3, 4, 5, 6], "so": [0, 2], "can": [0, 1, 2, 3, 4, 5], "our": [0, 1, 2, 3], "print": [0, 1, 2, 4, 5, 6], "zmin": [0, 1], "amin": [0, 1], "zmax": [0, 1], "amax": [0, 1], "size": [0, 2, 3, 4, 5, 6], "futur": [0, 1], "set": [0, 2, 3, 6], "14": [0, 1, 2, 4, 5, 6], "599803525171698": [0, 1], "1194": [0, 1], "4772333054245": [0, 1], "696": [0, 1], "creat": [0, 1], "To": [0, 1, 4, 5], "readabl": [0, 1], "replac": [0, 1], "have": [0, 1, 2], "extrem": [0, 1], "high": [0, 1], "nan": [0, 1], "number": [0, 1, 3], "same": [0, 1], "shape": [0, 1, 2, 3, 4, 6], "help": [0, 1], "ignor": [0, 1], "region": [0, 1], "ar": [0, 1, 2, 3, 4, 5], "interest": [0, 1], "fig": [0, 1, 2, 3, 4, 5, 6], "figur": [0, 1], "colorscal": [0, 1], "rainbow": [0, 1], "cmin": [0, 1], "cmax": [0, 1], "9": [0, 1, 2, 3, 4, 5, 6], "update_trac": [0, 1], "contours_z": [0, 1], "dict": [0, 1], "show": [0, 1, 3], "true": [0, 1, 2, 3, 4, 5, 6], "project_z": [0, 1], "update_layout": [0, 1], "titl": [0, 1, 4, 5, 6], "mueller": 0, "width": [0, 1, 2, 5], "500": [0, 1, 2, 4, 5, 6], "height": [0, 1], "scene": [0, 1], "zaxi": [0, 1], "dtick": [0, 1], "camera_ey": [0, 1], "becaus": [0, 1, 2], "cell": [0, 1, 3, 5], "accur": [0, 1, 2], "reflect": [0, 1], "copi": [0, 1, 6], "clean_z": [0, 1], "figsiz": [0, 1, 2, 4, 5, 6], "dpi": [0, 1, 2, 4, 5, 6], "150": [0, 1], "level": [0, 1], "12": [0, 1, 5], "ct": [0, 1], "color": [0, 1, 2, 3, 4, 5, 6], "clabel": [0, 1], "inlin": [0, 1], "fmt": [0, 1], "0f": [0, 1], "fontsiz": [0, 1], "contourf": [0, 1], "cmap": [0, 1], "cm": [0, 1], "extend": [0, 1, 3], "both": [0, 1], "vmin": [0, 1], "vmax": [0, 1], "xlabel": [0, 1, 2, 4, 5], "labelpad": [0, 1], "75": [0, 1], "ylabel": [0, 1, 2, 4, 5], "tick_param": [0, 1], "axi": [0, 1, 2, 4, 6], "pad": [0, 1], "labels": [0, 1], "cbar": [0, 1], "colorbar": [0, 1], "ax": [0, 1, 2, 3, 4, 5, 6], "m\u00fceller": [0, 1], "tight_layout": [0, 1, 2, 4, 5, 6], "after": [0, 3], "instal": [0, 1, 3, 4, 5, 6], "save": [0, 4, 5], "tensor": [0, 1, 2, 3, 4, 5, 6], "torch": [0, 1, 2, 3, 4, 5, 6], "f": [0, 1, 2, 3, 4, 5, 6], "util": [0, 2, 3, 4, 5, 6], "tensordataset": [0, 2, 3, 4, 5, 6], "dataload": [0, 2, 3, 4, 5, 6], "random_split": [0, 2, 3, 4, 5, 6], "dataset": [0, 2, 3, 4, 5], "train_load": [0, 2, 3, 4, 5], "batch_siz": [0, 2, 3, 4, 5], "32": [0, 1, 2, 3, 4, 5, 6], "shuffl": [0, 1], "below": [0, 1, 3], "schemat": [0, 3], "input": [0, 1, 2, 3, 4, 6], "given": [0, 1, 3], "weight": [0, 2, 3, 4], "w": [0, 2, 3, 4], "each": [0, 1, 2, 3], "neuron": [0, 2, 3, 4, 5], "hidden": 0, "layer": [0, 2, 3, 4], "bia": [0, 2, 3, 4], "ad": 0, "activ": [0, 2, 3, 4], "The": [0, 1, 2, 4, 5], "decid": 0, "bias": 0, "output": [0, 1, 2, 3, 4, 6], "produc": [0, 2, 3], "pred": [0, 6], "python": [0, 1, 3], "_loop": 0, "loop": 0, "through": [0, 2], "neuralnetwork": 0, "modul": [0, 2, 3, 4, 5, 6], "__init__": [0, 1, 2, 3, 4, 5, 6], "self": [0, 1, 2, 3, 4, 5, 6], "n1": [0, 1], "20": [0, 1, 2, 3, 4, 5], "super": [0, 1, 2, 3, 4, 5, 6], "sequenti": [0, 3, 4], "linear": 0, "one": 0, "forward": [0, 1, 2, 3, 4, 5, 6], "train_loop": 0, "optim": [0, 1, 2, 3, 4, 5, 6], "i_epoch": 0, "batch": [0, 2, 3, 4, 5], "enumer": 0, "comput": 0, "loss": [0, 1, 2, 3, 4, 5, 6], "mse_loss": [0, 2, 3, 4, 5], "squeez": [0, 1], "backpropag": [0, 1], "gradient": [0, 2, 3, 4, 5, 6], "updat": [0, 1], "zero_grad": [0, 1, 6], "zero": [0, 1, 6], "out": [0, 1], "previou": [0, 1, 6], "iter": [0, 1, 2, 6], "them": 0, "backward": [0, 1, 6], "step": [0, 1, 2, 4, 5, 6], "current": 0, "item": [0, 1, 6], "epoch": [0, 2, 3, 4, 5], "3d": [0, 1], "3f": [0, 1, 2, 3, 4, 5, 6], "5d": 0, "finish": 0, "when": 0, "desir": [0, 1], "ha": 0, "been": 0, "reach": [0, 1], "also": [0, 1, 2], "some": [0, 2, 4, 5], "term": [0, 2], "pass": 0, "entir": [0, 3], "rate": [0, 1], "determin": 0, "try": 0, "minim": 0, "faster": 0, "would": [0, 2], "larger": [0, 3], "stochast": 0, "descent": 0, "sgd": [0, 1], "algorithm": 0, "learning_r": [0, 2, 3, 4, 5, 6], "1e": 0, "1000": 0, "loss_fn": 0, "lr": [0, 1, 2, 3, 4, 5, 6], "t": [0, 1, 2, 3, 4, 5], "note": [0, 3], "exampl": [0, 2], "broken": 0, "21": [0, 5], "672": 0, "That": 0, "mean": [0, 1, 2, 3, 4, 5, 6], "an": [0, 1, 2, 3], "extra": 0, "24": 0, "togeth": [0, 3], "give": [0, 2, 3], "full": 0, "done": [0, 3], "28": [0, 1], "458": 0, "30": [0, 1], "642": 0, "480": 0, "301": 0, "472": 0, "094": 0, "009": 0, "300": [0, 1, 2, 4, 5, 6], "838": 0, "210": 0, "400": [0, 1], "723": 0, "278": 0, "708": 0, "827": 0, "600": [0, 1], "915": 0, "142": 0, "700": 0, "759": 0, "714": 0, "800": 0, "830": 0, "369": [0, 1], "900": 0, "464": 0, "387": 0, "final": [0, 2, 3], "analyt": [0, 1], "between": [0, 1, 2, 3], "show_surfac": [0, 1], "z_pred": [0, 1], "detach": [0, 1, 2, 3, 4, 5, 6], "zdiff": [0, 1], "subtract": [0, 1], "subplot": [0, 1, 2, 3, 4, 5, 6], "diff": 0, "min": [0, 1, 2], "max": [0, 1, 2, 4, 5, 6], "1184": 0, "2720674423874": 0, "299139974504966": 0, "order": 0, "closer": 0, "plug": 0, "name": [0, 1, 2, 5], "param": [0, 1, 2], "named_paramet": [0, 1], "weights0": 0, "elif": [0, 2, 3, 4], "bias0": 0, "weights2": 0, "bias2": 0, "xy0": 0, "z0": 0, "v1": 0, "i": [0, 1, 2, 3, 4, 5, 6], "v2": 0, "second": 0, "in_featur": [0, 2, 3, 4], "out_featur": [0, 2, 3, 4], "2185": 0, "grad_fn": [0, 1], "addbackward0": 0, "218454270765239": 0, "profession": 0, "infrom": 0, "logs_csv": [0, 2, 3, 4, 5], "directori": [0, 3], "pip": [0, 1, 2, 3, 4, 5, 6], "dev": [0, 2, 3, 4, 5, 6], "null": [0, 2, 3, 4, 5, 6], "pytorch_lightn": [0, 2, 3, 4, 5, 6], "pl": [0, 2, 3, 4, 5, 6], "logger": [0, 2, 3, 4, 5, 6], "pl_logger": [0, 2, 3, 4, 5, 6], "1m": 0, "0m": 0, "34": 0, "49mnotic": 0, "39": 0, "49m": 0, "new": [0, 1], "releas": 0, "avail": [0, 2], "31": [0, 2, 4, 5, 6], "49m22": 0, "49m23": 0, "run": [0, 1], "49mpip": 0, "upgrad": 0, "you": [0, 1], "mai": 0, "restart": 0, "kernel": [0, 6], "packag": [0, 1, 3], "captur": [0, 1, 2, 3, 4, 5, 6], "pl_model": 0, "lightningmodul": [0, 2, 3, 4, 5], "training_step": [0, 2, 3, 4, 5], "batch_idx": [0, 2, 3, 4, 5], "log": [0, 1, 2, 3, 4, 5, 6], "train_loss": [0, 2, 3, 4, 5], "configure_optim": [0, 2, 3, 4, 5], "adam": [0, 1, 2, 3, 4, 5, 6], "optimz": 0, "instead": [0, 2], "schedul": [0, 2, 3, 4, 5], "lr_schedul": [0, 2, 3, 4, 5], "exponentiallr": [0, 2, 3, 4, 5], "95": [0, 2, 3, 4, 5, 6], "interv": [0, 2, 3, 4, 5], "frequenc": [0, 2, 3, 4, 5], "csv_logger": [0, 2, 3, 4, 5], "csvlogger": [0, 2, 3, 4, 5], "trainer": [0, 2, 3, 4, 5], "max_epoch": [0, 2, 3, 4, 5], "gpu": [0, 2, 4, 5], "fals": [0, 1, 2, 3, 4], "tpu": [0, 2], "core": [0, 2], "ipu": [0, 2], "hpu": [0, 2], "miss": 0, "folder": 0, "lightning_log": [0, 2, 3, 4, 5], "type": [0, 2, 3, 4, 5, 6], "81": 0, "trainabl": [0, 2], "non": [0, 2], "total": [0, 2, 3, 5], "000": [0, 1], "estim": [0, 2], "mb": [0, 2], "increas": [0, 1], "panda": [0, 2, 3, 4, 5], "pd": [0, 2, 3, 4, 5], "read_csv": [0, 2, 3, 4, 5], "version_0": [0, 2, 5], "metric": [0, 2, 3, 4, 5], "csv": [0, 2, 3, 4, 5], "semilog": [0, 2, 3, 4, 5], "set_xlabel": [0, 2, 3, 4, 5, 6], "set_ylabel": [0, 2, 3, 4, 5, 6], "text": [0, 3], "again": 0, "1152": 0, "682101622075": 0, "644299219725138": 0, "start": 1, "function": [1, 3, 5, 7], "deriv": 1, "respect": [1, 3], "muller": 1, "frac": [1, 2, 3], "dv": 1, "dx": 1, "2a_k": 1, "dy": 1, "A": [1, 2, 4, 5], "dx_valu": 1, "dy_valu": 1, "neural": [1, 7], "network": [1, 5, 7], "displai": [1, 4, 5, 6], "test": [1, 6], "dx_truncat": 1, "dy_trunc": 1, "dvx": 1, "dvy": 1, "ml": 1, "x_truncat": 1, "y_truncat": 1, "x_ref": 1, "y_ref": 1, "z_ref": 1, "896": 1, "more": 1, "let": 1, "variabl": [1, 6], "vector": [1, 2, 6], "featur": [1, 4, 5, 6], "textbf": 1, "n": [1, 3], "configur": 1, "assembl": 1, "_1": 1, "_n": 1, "correspond": 1, "observ": [1, 2], "y_1": 1, "y_n": 1, "noisi": 1, "sampl": 1, "assum": 1, "seper": 1, "underli": 1, "accord": 1, "mathit": 1, "\u03b5": 1, "where": [1, 2, 3], "nois": [1, 6], "follow": [1, 2, 4, 5], "distribut": [1, 2, 6], "sim": 1, "mathcal": [1, 3], "\u03c3": 1, "2_n": 1, "sigma": 1, "paramet": [1, 2, 3, 4, 5, 6], "prior": 1, "covarainc": 1, "matrix": [1, 6], "covari": 1, "base": 1, "simular": 1, "begin": [1, 2, 3], "bmatrix": 1, "ldot": 1, "vdot": 1, "ddot": 1, "end": [1, 2, 3], "radial": [1, 2], "basi": 1, "_a": 1, "_b": 1, "2_f": 1, "2l": 1, "vertic": 1, "variatiton": 1, "l": 1, "length": [1, 2], "setup": 1, "taken": 1, "directli": 1, "minor": 1, "chang": [1, 2], "class": [1, 4, 5, 6], "exactgpmodel": 1, "exactgp": [1, 6], "train_x": 1, "train_i": 1, "likelihood": [1, 6], "mean_modul": [1, 6], "zeromean": 1, "covar_modul": [1, 6], "scalekernel": [1, 6], "rbfkernel": [1, 6], "mean_x": [1, 6], "covar_x": [1, 6], "multivariatenorm": [1, 6], "x_gpr": 1, "z_gpr": 1, "initi": [1, 6], "gaussianlikelihood": [1, 6], "With": 1, "variat": 1, "sai": 1, "theta": 1, "maxim": 1, "margin": [1, 6], "p": 1, "\u03b8": 1, "pi": [1, 2, 3, 4, 6], "demonstr": 1, "neg": 1, "smooth": [1, 7], "fix": 1, "grid": 1, "search": 1, "over": 1, "noise_valu": 1, "scale_and_length": 1, "j": [1, 2, 3], "50": [1, 3, 5], "x_plt": 1, "y_plt": 1, "z_plt": 1, "pair": 1, "hyper": 1, "noise_covar": 1, "base_kernel": [1, 6], "lengthscal": [1, 6], "outputscal": [1, 6], "mll": [1, 6], "exactmarginalloglikelihood": [1, 6], "tricontour": 1, "tricontourf": 1, "set_label": 1, "rotat": [1, 2, 3], "270": 1, "previous": [1, 2, 3, 4, 5], "built": 1, "until": 1, "Then": 1, "train_model": 1, "x_train": [1, 6], "y_train": 1, "print_hp": 1, "untrain": 1, "param_nam": 1, "42": 1, "5f": 1, "training_it": [1, 6], "find": [1, 6], "gp": [1, 6], "calcul": [1, 2], "raw_nois": 1, "54117": 1, "raw_outputscal": 1, "54132": 1, "raw_lengthscal": 1, "879": 1, "263": 1, "046": 1, "787": 1, "136": 1, "360": 1, "087": 1, "633": 1, "209": 1, "40": 1, "854": 1, "119": 1, "536": 1, "250": 1, "479": 1, "149": 1, "459": 1, "272": 1, "60": 1, "255": 1, "178": 1, "405": [1, 3], "279": 1, "70": 1, "135": 1, "204": 1, "80": 1, "063": 1, "230": 1, "345": 1, "275": 1, "90": 1, "014": 1, "254": 1, "327": 1, "268": 1, "980": 1, "313": 1, "260": 1, "91370": 1, "98492": 1, "03970": 1, "space": [1, 2], "\u03bc": 1, "abov": 1, "equaiton": 1, "eval": [1, 6], "z_var": 1, "47233": 1, "96655": 1, "08929": 1, "30226": 1, "recal": 1, "lesson": 1, "compos": 1, "depend": [1, 3], "upon": 1, "well": 1, "partial": 1, "themselv": 1, "consequ": 1, "thei": 1, "incorpor": 1, "target": 1, "explicit": 1, "_": [1, 2, 3, 4, 6], "ext": 1, "account": 1, "addit": 1, "displaystyl": 1, "gpmodelwithderiv": 1, "constantmeangrad": 1, "rbfkernelgrad": 1, "ard_num_dim": 1, "multitaskmultivariatenorm": 1, "dx_gpr": 1, "dy_gpr": 1, "y_gpr": 1, "stack": [1, 3], "includ": [1, 6], "x_test": [1, 6], "y_test": 1, "flatten": 1, "multitaskgaussianlikelihood": 1, "num_task": 1, "use_gpu": 1, "cuda": [1, 2, 4, 5, 6], "is_avail": [1, 2], "textrm": [1, 2], "As": 1, "which": [1, 2, 3], "contain": [1, 2], "m": 1, "19": 1, "532": [1, 2], "693": 1, "613": 1, "744": 1, "644": 1, "410": 1, "798": 1, "598": 1, "797": 1, "884": 1, "852": 1, "556": 1, "555": 1, "850": 1, "829": 1, "908": 1, "516": 1, "515": 1, "903": 1, "108": 1, "965": 1, "478": 1, "955": 1, "738": 1, "021": 1, "446": 1, "444": 1, "003": 1, "751": 1, "078": 1, "416": 1, "412": 1, "049": 1, "089": 1, "133": 1, "388": 1, "383": [1, 2], "090": 1, "659": 1, "188": 1, "364": 1, "358": 1, "128": 1, "its": 1, "expect": 1, "e": [1, 2, 3], "2_": 1, "ast_": 1, "ast": 1, "associ": 1, "var": 1, "q": [1, 6], "bigg": 1, "no_grad": 1, "fast_comput": 1, "log_prob": 1, "covar_root_decomposit": 1, "cpu": [1, 2, 4, 5, 6], "n2": 1, "dx_pred": 1, "dx_diff": 1, "dy_pr": 1, "dy_diff": 1, "work": 1, "look": 1, "accuraci": 1, "root": 1, "squar": 1, "error": [1, 2, 4], "rmse": [1, 6], "decreas": 1, "r": [1, 2, 3, 4, 5, 6], "tabul": 1, "z_test": 1, "evaluate_model": 1, "train_z": 1, "test_x": 1, "test_z": 1, "preds_train": 1, "preds_test": 1, "rmse_train": 1, "sqrt": [1, 2, 3, 4, 5, 6], "rmse_test": 1, "r2": 1, "sum": [1, 2, 3, 4, 5, 6], "q2": [1, 6], "reduce_training_set": 1, "new_siz": 1, "arr_index": 1, "random": 1, "new_model": 1, "size_list": 1, "rmse_train_list": 1, "r2_list": 1, "rmse_test_list": 1, "q2_list": 1, "training_set_s": 1, "set_siz": 1, "training_set_dict": 1, "header": 1, "kei": 1, "floatfmt": 1, "4f": 1, "opt": [1, 3], "hostedtoolcach": [1, 3], "16": [1, 2, 3, 4, 6], "x64": [1, 3], "lib": [1, 3], "python3": [1, 3], "site": [1, 3], "exact_gp": 1, "py": [1, 3, 4, 5, 6], "283": 1, "gpinputwarn": 1, "match": 1, "did": 1, "forget": 1, "call": [1, 2, 3, 5], "0466": 1, "9756": 1, "0517": 1, "9724": 1, "0476": 1, "0581": 1, "9675": 1, "0612": 1, "0707": 1, "9603": 1, "0634": 1, "0783": 1, "9481": 1, "1461": 1, "0957": 1, "9365": 1, "1360": 1, "0770": 1, "8895": 1, "2094": 1, "result": [1, 2], "befor": 1, "rbf": 1, "matern": 1, "ration": 1, "quadrat": 1, "rq": 1, "gpmodel_kernel": 1, "constantmean": [1, 6], "maternkernel": 1, "rqkernel": 1, "train_model_k": 1, "calc": [1, 6], "backprop": [1, 6], "kernel_nam": 1, "str": [1, 2, 3, 4], "split": 1, "0031": 1, "sqrtbackward0": 1, "0026": 1, "0021": 1, "atom": [2, 3], "center": 2, "allow": 2, "ensur": [2, 3], "energi": [2, 3, 4, 5, 6], "invari": 2, "translat": [2, 3], "train": [2, 4, 5, 6], "machin": 2, "coordin": [2, 3, 5], "qm_coord": [2, 3, 4, 5, 6], "npy": [2, 3, 4, 5, 6], "1800": 2, "3": [2, 3, 4, 5, 6], "qm_elem": [2, 4, 5, 6], "txt": [2, 4, 5, 6], "pm3": [2, 4, 5, 6], "energy_sqm": [2, 4, 5, 6], "qm_grad_sqm": [2, 4, 5, 6], "b3lyp": [2, 4, 5, 6], "g": [2, 3, 4, 5, 6], "qm_grad": [2, 4, 5, 6], "These": 2, "provid": 2, "semi": 2, "empir": 2, "qm": 2, "sequenc": [2, 3, 4, 5, 6], "tupl": [2, 3, 4, 5, 6], "nn": [2, 3, 4, 5, 6], "far": 2, "cartesian": 2, "howev": 2, "ideal": 2, "molecul": [2, 3], "case": [2, 3], "chemistri": 2, "impli": 2, "s": [2, 3, 4, 5], "incorrect": 2, "describ": [2, 3], "newli": 2, "transform": [2, 3], "2007": 2, "paper": [2, 3], "cite": 2, "sever": 2, "cutoff": [2, 3], "were": 2, "introduc": 2, "align": 2, "f_c": 2, "r_": [2, 3], "ij": 2, "cl": 2, "time": [2, 3, 4, 5], "co": [2, 3, 4, 6], "r_c": [2, 3], "le": 2, "gt": 2, "distanc": [2, 3], "g_i": 2, "neq": 2, "all": 2, "eta": 2, "gaussian": [2, 7], "shift": 2, "peak": 2, "angular": 2, "zeta": [2, 4, 6], "lambda": 2, "theta_": 2, "ijk": [2, 4, 6], "ik": 2, "jk": 2, "mathbf": 2, "cdot": [2, 3], "explor": 2, "environ": 2, "posit": 2, "2017": 2, "isayev": 2, "roitberg": 2, "develop": 2, "anakin": 2, "me": 2, "engin": 2, "molecular": 2, "refer": [2, 3, 4, 5, 6], "modifi": 2, "shell": 2, "probe": 2, "differ": 2, "angl": 2, "pairwise_vector": [2, 4, 6], "symmetry_function_g2": [2, 4, 6], "bp": 2, "symmetry_function_g3": [2, 4, 6], "symmetry_function_g3ani": [2, 4, 6], "coord": [2, 3, 4, 5, 6], "num_batch": [2, 3, 4, 6], "num_channel": [2, 3, 4, 6], "rij": [2, 3, 4, 6], "none": [2, 3, 4, 5, 6], "mask": [2, 3, 4, 6], "ey": [2, 3, 4, 6], "dtype": [2, 3, 4, 5, 6], "bool": [2, 3, 4, 6], "devic": [2, 4, 6], "remov": [2, 3, 4, 6], "interact": [2, 3, 4, 6], "masked_select": [2, 3, 4, 6], "unsqueez": [2, 3, 4, 6], "view": [2, 3, 4, 6], "rcr": [2, 4, 6], "float": [2, 4, 6], "etar": [2, 4, 6], "shfr": [2, 4, 6], "dij": [2, 3, 4, 6], "norm": [2, 3, 4, 6], "dim": [2, 3, 4, 5, 6], "fij": [2, 4, 6], "g2": [2, 4, 6], "rca": [2, 4, 6], "etaa": [2, 4, 6], "combin": [2, 4, 6], "r12": [2, 4, 6], "r13": [2, 4, 6], "r23": [2, 4, 6], "d12": [2, 4, 6], "d13": [2, 4, 6], "d23": [2, 4, 6], "f12": [2, 4, 6], "f13": [2, 4, 6], "f23": [2, 4, 6], "cosin": [2, 4, 6], "einsum": [2, 4, 6], "ijkl": [2, 4, 6], "g3": [2, 4, 6], "shfz": [2, 4, 6], "shfa": [2, 4, 6], "aco": [2, 4, 6], "feed": 2, "aspect": 2, "connect": 2, "int": [2, 3, 4, 5, 6], "residu": [2, 3, 4], "els": [2, 3, 4, 6], "register_paramet": [2, 3, 4], "reset_paramet": [2, 3, 4], "init": [2, 3, 4], "kaiming_uniform_": [2, 3, 4], "zip": [2, 3, 4], "fan_in": [2, 3, 4], "_calculate_fan_in_and_fan_out": [2, 3, 4], "bound": [2, 3, 4], "uniform_": [2, 3, 4], "channel": [2, 3, 4], "bmm": [2, 3, 4], "transpos": [2, 3, 4], "cat": [2, 3, 4], "rais": [2, 3, 4], "notimplementederror": [2, 3, 4], "Not": [2, 3, 4], "implement": [2, 4, 7], "extra_repr": [2, 3, 4], "format": [2, 3, 4], "take": 2, "inform": 2, "characterist": 2, "uniqu": [2, 4, 5, 6], "assert": [2, 4, 6], "atom_typ": [2, 3, 4, 5, 6], "concat": [2, 4, 6], "properti": [2, 3, 4, 6], "output_length": [2, 3, 4, 5, 6], "similar": 2, "featureani": [2, 4, 6], "n_type": [2, 3, 4], "240": [2, 3, 4], "fitting_net": [2, 3, 4, 5], "descriptor": [2, 3, 4, 5, 6], "5e": [2, 3, 4, 5, 6], "requires_grad_": [2, 3, 4, 5], "atomic_energi": [2, 3, 4, 5], "unbind": [2, 3, 4, 5], "autograd": [2, 3, 4, 5, 6], "grad": [2, 3, 4, 5, 6], "create_graph": [2, 3, 4, 5], "hstack": [2, 3, 4, 5], "ene_pr": [2, 3, 4, 5], "grad_pr": [2, 3, 4, 5], "predict": [2, 3, 4, 5, 6], "ene_loss": [2, 4, 5], "grad_loss": [2, 4, 5], "param_group": [2, 4, 5], "start_lr": [2, 4, 5], "initial_lr": [2, 4, 5], "w_ene": [2, 4, 5], "w_grad": [2, 4, 5], "99": [2, 4, 5], "l2_trn": [2, 4, 5], "l2_e_trn": [2, 4, 5], "l2_f_trn": [2, 4, 5], "validation_step": [2, 4, 5], "set_grad_en": [2, 4, 5], "val_loss": [2, 4, 5], "l2_tst": [2, 4, 5], "l2_e_tst": [2, 4, 5], "l2_f_tst": [2, 4, 5], "ds": 2, "datasourc": 2, "open": [2, 3], "http": [2, 3, 4, 5, 6], "com": [2, 4, 5, 6], "cc": [2, 4, 5, 6], "ats": [2, 4, 5, 6], "raw": [2, 3, 4, 5, 6], "main": [2, 4, 5, 6], "butan": 2, "rb": [2, 3], "float32": [2, 3, 4, 5, 6], "loadtxt": [2, 3, 4, 5, 6], "elem": [2, 4, 5, 6], "tolist": [2, 4, 5, 6], "index": [2, 4, 5, 6], "repeat": [2, 3, 4, 5, 6], "27": [2, 4, 5, 6], "2114": [2, 4, 5, 6], "23": [2, 4, 5, 6], "061": [2, 4, 5, 6], "qm_gradient": [2, 4, 5, 6], "529177249": [2, 4, 5, 6], "from_numpi": [2, 3, 4, 5, 6], "seed_everyth": [2, 4, 5], "val": [2, 4, 5], "1728": 2, "72": 2, "val_load": [2, 4, 5], "global": 2, "seed": 2, "torchani": [2, 4, 6], "2000e": [2, 4, 6], "00": [2, 4, 6], "5000e": [2, 4, 6], "6000000e": [2, 4, 6], "01": [2, 4, 6], "0000000e": [2, 4, 6], "1687500e": [2, 4, 6], "4375000e": [2, 4, 6], "7062500e": [2, 4, 6], "9750000e": [2, 4, 6], "2437500e": [2, 4, 6], "5125000e": [2, 4, 6], "7812500e": [2, 4, 6], "0500000e": [2, 4, 6], "3187500e": [2, 4, 6], "5875000e": [2, 4, 6], "8562500e": [2, 4, 6], "1250000e": [2, 4, 6], "3937500e": [2, 4, 6], "6625000e": [2, 4, 6], "9312500e": [2, 4, 6], "2000000e": [2, 4, 6], "9634954e": [2, 4, 6], "8904862e": [2, 4, 6], "8174770e": [2, 4, 6], "3744468e": [2, 4, 6], "7671459e": [2, 4, 6], "1598449e": [2, 4, 6], "5525440e": [2, 4, 6], "9452431e": [2, 4, 6], "5500000e": [2, 4, 6], "8500000e": [2, 4, 6], "h": [2, 4, 6], "o": [2, 4, 6], "04": [2, 4, 6], "71": [2, 4, 6], "79": [2, 4, 6], "log_every_n_step": [2, 4, 5], "acceler": [2, 4, 5], "auto": 2, "pt": [2, 4, 5], "model_script": [2, 4, 5], "jit": [2, 4, 5], "attempt": [2, 4, 5], "preserv": [2, 3, 4, 5], "behavior": [2, 4, 5], "oper": [2, 4, 5], "across": [2, 4, 5], "version": [2, 4, 5], "load_state_dict": [2, 4, 5], "state_dict": [2, 4, 5], "to_torchscript": [2, 4, 5], "cpu_coord": 2, "cpu_atom": 2, "e1": [2, 4, 5, 6], "e2": [2, 4, 5, 6], "plot": [2, 4, 5, 6], "linestyl": [2, 3, 4, 5, 6], "marker": [2, 3, 4, 5, 6], "concaten": [2, 4, 5, 6], "linewidth": [2, 3, 4, 5, 6], "kcal": [2, 3, 4, 5, 6], "mol": [2, 3, 4, 5, 6], "annot": [2, 4, 5, 6], "rmsd": [2, 4, 5, 6], "05": [2, 4, 5, 6], "xycoord": [2, 4, 5, 6], "fraction": [2, 4, 5, 6], "f1": [2, 3, 4, 5, 6], "f2": [2, 3, 4, 5, 6], "ab": [2, 4, 5, 6], "forc": [2, 4, 5, 6], "savefig": [2, 4, 5, 6], "png": [2, 4, 5, 6], "param_tensor": [2, 4, 5], "48": 2, "isnul": [2, 4, 5], "y2": [2, 4], "valid": [2, 4, 5], "label": [2, 4, 5], "legend": [2, 4, 5], "1350": 2, "0x7f54a3d0d060": 2, "xiaoliang": 3, "pan": 3, "2022": 3, "origin": 3, "symmetri": [3, 7], "inter": 3, "model": [3, 4, 5, 6], "finit": 3, "system": [3, 5], "linfeng": 3, "zhang": 3, "jiequn": 3, "han": 3, "wang": 3, "wissam": 3, "saidi": 3, "roberto": 3, "car": 3, "weinan": 3, "doi": 3, "org": 3, "48550": 3, "arxiv": 3, "1805": 3, "09003": 3, "requir": 3, "lightn": [3, 4, 5, 6], "build": 3, "envrion": 3, "turn": 3, "d_i": 3, "map": 3, "matric": 3, "manner": 3, "permut": 3, "obtain": 3, "extens": 3, "shown": 3, "_i": 3, "eqnarrai": 3, "pmatrix": 3, "1i": 3, "x_": 3, "y_": 3, "z_": 3, "2i": 3, "i1": 3, "ni": 3, "neighbor": 3, "ji": 3, "cs": 3, "down": 3, "becom": 3, "greater": 3, "than": 3, "approach": 3, "It": 3, "beyond": 3, "continu": 3, "differenti": 3, "_environ": 3, "local_environ": 3, "dij_inv": 3, "dij2_inv": 3, "loc_env_r": 3, "loc_env_a": 3, "gener": 3, "compon": 3, "while": 3, "schem": 3, "present": 3, "m_1": 3, "chemic": 3, "speci": 3, "two": 3, "dimens": 3, "i2": 3, "m_2": 3, "repres": 3, "By": 3, "multipli": 3, "form": [3, 4, 5, 6], "encod": 3, "d": [3, 6], "yield": 3, "1_i": 3, "2_i": 3, "construct": 3, "25": [3, 5], "axis_neuron": [3, 5], "local_embed": 3, "neighbor_typ": 3, "indic": 3, "load": [3, 4, 5, 6], "content": [3, 4, 5], "drive": [3, 4, 5], "mydriv": [3, 4, 5], "colab": [3, 5], "notebook": 3, "deeppot_pytorch": 3, "input_coord": 3, "int64": 3, "input_grad": 3, "120": 3, "filenotfounderror": 3, "traceback": [3, 5], "most": [3, 5], "recent": [3, 5], "last": [3, 5], "line": [3, 4, 5, 6], "file": [3, 4, 5, 6], "npyio": 3, "mmap_mod": 3, "allow_pickl": 3, "fix_import": 3, "max_header_s": 3, "403": 3, "own_fid": 3, "404": 3, "fid": 3, "enter_context": 3, "os_fspath": 3, "406": 3, "408": 3, "code": 3, "distinguish": 3, "binari": 3, "pickl": 3, "errno": 3, "No": [3, 5], "set_aspect": 3, "equal": 3, "adjust": 3, "box": 3, "18": 3, "version_5": 3, "reduct": 3, "data": [4, 5, 6], "github": [4, 5, 6], "mlp_class": [4, 5, 6], "mode": [4, 5, 6], "markdown": [4, 5, 6], "pytorch": [4, 5, 6, 7], "2100": [4, 5, 6], "rm": [4, 5, 6], "sample_data": [4, 5, 6], "wget": [4, 5, 6], "claisen_rearrang": [4, 5, 6], "usageerror": [4, 6], "magic": [4, 6], "found": [4, 6], "librari": [4, 5, 6], "dens": 4, "fit": [4, 5, 7], "bpnn": 4, "2016": [4, 5], "84": [4, 5], "ani": [4, 6], "model_diff": [4, 5], "model_diff_script": [4, 5], "version_10": 4, "f10_e500": [4, 5], "version_2": [4, 5], "trajectori": 5, "reaction": 5, "d_1": 5, "d_2": 5, "window": 5, "frame": 5, "ps": 5, "everi": 5, "fs": 5, "claisen": 5, "u": 5, "kora": 5, "img": 5, "mp4": 5, "upload_publ": 5, "url": 5, "ipython": 5, "html": 5, "video": 5, "src": 5, "control": 5, "githubusercont": 5, "pent": 5, "enal": 5, "200px": 5, "modulenotfounderror": 5, "get_ipython": 5, "ml_qmmm": 5, "deeppot": [5, 7], "googl": 5, "mount": 5, "gpytorch": 6, "bpgpr": 6, "nskip": 6, "hyperparamet": 6, "prepar": 6, "requires_grad": 6, "y_pred": 6, "y_mean": 6, "y_var": 6, "varianc": 6, "y_covar": 6, "covariance_matrix": 6, "ref": 6, "evalu": 6, "perform": 6, "y_rmse": 6, "y_q2": 6, "auto_grad": 6, "aa": 6, "qm_coord_train": 6, "process": 7, "regress": 7, "behler": 7, "parrinello": 7, "deep": 7, "edit": 7, "se": 7, "netwotk": 7}, "objects": {}, "objtypes": {}, "objnames": {}, "titleterms": {"fit": [0, 2, 3], "neural": [0, 2, 3, 4, 5], "network": [0, 2, 3], "model": [0, 1, 2, 7], "defin": [0, 1, 2, 3], "m\u00fcller": 0, "brown": [0, 1], "potenti": [0, 1, 3, 5, 7], "energi": [0, 1], "function": [0, 2, 4, 6], "gener": [0, 1], "train": [0, 1, 3], "data": [0, 1, 2, 3], "visual": [0, 1], "3": [0, 1], "d": [0, 1], "project": [0, 1], "surfac": [0, 1], "contour": [0, 1], "load": [0, 1, 2], "pytorch": [0, 1, 2, 3], "A": [0, 3, 7], "basic": 0, "class": [0, 2, 3, 7], "plot": [0, 1, 3], "refer": [0, 1], "predict": [0, 1], "differ": [0, 1], "take": 0, "look": 0, "nn": 0, "paramet": 0, "more": 0, "autom": 0, "refin": 0, "implement": [0, 3], "lightn": [0, 2], "error": [0, 3], "gaussian": [1, 6], "process": [1, 6], "regress": [1, 6], "mueller": 1, "gpytorch": 1, "gpr": 1, "learn": [1, 7], "hyperparamet": 1, "varianc": 1, "us": [1, 2], "gradient": 1, "perform": 1, "set": 1, "size": 1, "compar": 1, "kernel": 1, "behler": [2, 4, 6], "parrinello": [2, 4, 6], "symmetri": [2, 4, 6], "import": [2, 3], "librari": [2, 3], "from": 2, "github": 2, "mlp_class": 2, "instal": 2, "featur": [2, 3], "extract": [2, 3], "sequenti": 2, "dens": [2, 3], "creat": 2, "ani": 2, "bpnn": 2, "save": 2, "file": 2, "deep": [3, 5], "smooth": 3, "edit": 3, "deeppot": 3, "se": 3, "local": 3, "environ": 3, "embed": 3, "matrix": 3, "The": 3, "rmsd": 3, "forc": 3, "netwotk": [4, 5], "mlp": 7, "tabl": 7, "content": 7, "machin": 7, "b": 7, "molecular": 7, "represent": 7, "c": 7}, "envversion": {"sphinx.domains.c": 2, "sphinx.domains.changeset": 1, "sphinx.domains.citation": 1, "sphinx.domains.cpp": 6, "sphinx.domains.index": 1, "sphinx.domains.javascript": 2, "sphinx.domains.math": 2, "sphinx.domains.python": 3, "sphinx.domains.rst": 2, "sphinx.domains.std": 2, "sphinx.ext.intersphinx": 1, "sphinxcontrib.bibtex": 9, "sphinx": 56}})