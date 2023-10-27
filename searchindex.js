Search.setIndex({"docnames": ["Lesson1_FNN", "Lesson2_GPR", "Lesson3_BP", "Lesson4_DeepPot", "Lesson5_BP-FNN_MLP", "Lesson6_DeepPot-FNN_MLP", "Lesson7_BP-GPR_MLP", "index"], "filenames": ["Lesson1_FNN.ipynb", "Lesson2_GPR.ipynb", "Lesson3_BP.ipynb", "Lesson4_DeepPot.ipynb", "Lesson5_BP-FNN_MLP.ipynb", "Lesson6_DeepPot-FNN_MLP.ipynb", "Lesson7_BP-GPR_MLP.ipynb", "index.md"], "titles": ["<span class=\"section-number\">1. </span>Fitting Neural Network Models", "<span class=\"section-number\">2. </span>Gaussian Process Regression Models", "<span class=\"section-number\">3. </span>Behler-Parrinello  Symmetry Functions", "<span class=\"section-number\">4. </span>A PyTorch implementation of Deep Potential-Smooth Edition (DeepPot-SE)", "<span class=\"section-number\">5. </span>Behler-Parrinello Fitting Neural Network with Machine Learning Potential (BP-FNN MLP) Models for the Claisen Rearrangement", "<span class=\"section-number\">6. </span>Lesson 6: DeepPot-Smooth Edition Fitting Neural Network with Machine Learning Potentials (DeepPot-SE-FNN MLP)", "<span class=\"section-number\">7. </span>Lesson 7: Behler-Parrinello Gaussian Process Regression (BP-GPR) for Machine Learning Potentials", "MLP Tutorial"], "terms": {"In": [0, 1, 2], "thi": [0, 1, 2, 3, 4, 5, 6], "tutori": [0, 1, 2, 4, 5, 6], "we": [0, 1, 2, 3, 4, 5, 6], "learn": [0, 2, 3], "how": [0, 1], "us": [0, 3, 5], "point": [0, 1], "For": [0, 1, 4, 5, 6], "definit": [0, 1], "see": [0, 1, 3], "here": [0, 1, 2, 3, 4, 5, 6], "v": [0, 1], "x": [0, 1, 2, 3, 4, 5, 6], "y": [0, 1, 2, 4, 5, 6], "sum_": [0, 1, 2], "k": [0, 1, 2, 3, 4, 5, 6], "0": [0, 1, 2, 3, 4, 5, 6], "a_k": [0, 1], "mathrm": [0, 1], "exp": [0, 1, 2, 4, 6], "left": [0, 1, 2, 3, 5], "x_k": [0, 1], "2": [0, 1, 2, 3, 4, 5, 6], "b_k": [0, 1], "y_k": [0, 1], "c_k": [0, 1], "right": [0, 1, 2, 3, 5], "from": [0, 1, 2, 3, 4, 5, 6], "math": [0, 1, 2, 3, 4, 5, 6], "import": [0, 1], "pow": [0, 1], "def": [0, 1, 2, 3, 4, 5, 6], "mueller_brown_potenti": [0, 1], "200": [0, 1, 6], "100": [0, 1, 3, 4, 5, 6], "170": [0, 1], "15": [0, 1], "1": [0, 1, 2, 3, 4, 5, 6], "6": [0, 1, 2, 3, 4, 6, 7], "5": [0, 1, 2, 3, 4, 5, 6], "7": [0, 1, 7], "b": [0, 1, 2, 3, 4, 5], "11": [0, 1], "c": [0, 1, 2, 3, 4, 5, 6], "10": [0, 1, 2, 3, 4, 5, 6], "x0": [0, 1], "y0": [0, 1], "z": [0, 1], "rang": [0, 1, 2, 3, 4, 5, 6], "4": [0, 1, 2, 3, 4, 5, 6], "scale": [0, 1, 3, 5], "make": [0, 1, 2, 4, 5, 6], "easier": [0, 1], "return": [0, 1, 2, 3, 4, 5, 6], "first": [0, 1, 3, 4, 5, 6], "need": [0, 1, 3, 4, 5, 6], "numpi": [0, 1, 2, 3, 4, 5, 6], "np": [0, 1, 2, 3, 4, 5, 6], "grid": 0, "x_rang": 0, "arang": [0, 1, 2, 4, 6], "8": [0, 1, 2, 4, 6], "dtype": [0, 2, 3, 4, 5, 6], "float32": [0, 2, 3, 4, 5, 6], "y_rang": 0, "meshgrid": [0, 1, 2, 4, 6], "comput": 0, "each": [0, 1, 2, 3, 4, 5, 6], "mueller_brown_potential_vector": 0, "vector": [0, 1, 2, 6], "otyp": 0, "keep": [0, 1], "onli": [0, 1, 2], "low": [0, 1], "train_mask": 0, "x_train": [0, 1, 6], "y_train": [0, 1], "z_train": 0, "print": [0, 1, 2, 4, 5, 6], "f": [0, 1, 2, 3, 4, 5, 6], "z_min": 0, "min": [0, 1, 2, 3, 4, 5], "z_max": 0, "max": [0, 1, 2, 3, 4, 5, 6], "size": [0, 2, 3, 4, 5, 6], "futur": [0, 1], "set": [0, 2, 3, 4, 5, 6], "len": [0, 1, 2, 3, 4, 5, 6], "14": [0, 1, 2, 3, 4, 5, 6], "599802017211914": 0, "1194": [0, 1], "4622802734375": 0, "696": [0, 1], "now": [0, 1, 2, 3, 4, 5, 6], "creat": [0, 1], "our": [0, 1, 2, 3, 4, 5, 6], "ar": [0, 1, 2, 3, 4, 5, 6], "go": [0, 1], "plotli": [0, 1], "librari": 0, "an": [0, 1, 2, 3, 5], "interact": [0, 2, 3, 4, 5, 6], "part": 0, "below": [0, 1, 2, 3, 4, 5], "arbitrari": 0, "unit": 0, "actual": 0, "graph_object": [0, 1], "fig": [0, 1, 2, 3, 4, 5, 6], "figur": [0, 1], "colorscal": [0, 1], "rainbow": [0, 1], "cmin": [0, 1], "cmax": [0, 1], "9": [0, 1, 2, 4, 5, 6], "update_trac": [0, 1], "contours_z": [0, 1], "dict": [0, 1], "show": [0, 1, 2, 3], "true": [0, 1, 2, 3, 4, 5, 6], "project_z": [0, 1], "update_layout": [0, 1], "titl": [0, 1, 2], "mueller": [0, 2], "width": [0, 1, 2, 4, 5], "500": [0, 1, 2, 3, 6], "height": [0, 1], "scene": [0, 1], "zaxi": [0, 1], "dtick": [0, 1], "camera_ey": [0, 1], "sinc": 0, "similar": [0, 2, 4, 6], "later": 0, "do": 0, "matplotlib": [0, 1, 2, 3, 4, 5, 6], "pyplot": [0, 1, 2, 3, 4, 5, 6], "plt": [0, 1, 2, 3, 4, 5, 6], "plot_contour_map": 0, "level": [0, 1, 4, 5, 6], "12": [0, 1, 2], "vmin": [0, 1], "vmax": [0, 1], "figsiz": [0, 1, 2, 4, 5, 6], "dpi": [0, 1, 2, 4, 5, 6], "150": [0, 1], "ct": [0, 1], "color": [0, 1, 2, 3, 4, 5, 6], "clabel": [0, 1], "inlin": [0, 1], "fmt": [0, 1], "0f": [0, 1], "fontsiz": [0, 1, 2], "contourf": [0, 1], "cmap": [0, 1], "extend": [0, 1, 3], "both": [0, 1], "xlabel": [0, 1, 2, 4, 5], "labelpad": [0, 1], "75": [0, 1], "ylabel": [0, 1, 2, 4, 5], "tick_param": [0, 1], "axi": [0, 1, 2, 3, 4, 5, 6], "pad": [0, 1], "labels": [0, 1], "cbar": [0, 1], "colorbar": [0, 1], "ax": [0, 1, 2, 3, 4, 5, 6], "tight_layout": [0, 1, 2, 4, 5, 6], "schemat": [0, 3, 5], "input": [0, 1, 2, 3, 4, 5, 6], "given": [0, 1, 3, 5], "weight": [0, 3], "w": [0, 2, 3, 4, 5], "neuron": [0, 2, 3, 4, 5], "hidden": 0, "layer": [0, 2, 3, 4, 5], "bia": [0, 2, 3, 4, 5], "ad": 0, "valu": [0, 1, 2, 3, 4, 5, 6], "activ": [0, 2, 3, 4, 5], "tanh": [0, 1, 2, 3, 4, 5], "The": [0, 1, 6], "decid": 0, "bias": 0, "output": [0, 1, 2, 3, 4, 5, 6], "produc": [0, 2, 3], "pred": [0, 6], "python": [0, 1], "_loop": 0, "loop": 0, "through": [0, 2], "torch": [0, 1, 2, 3, 4, 5, 6], "default": 0, "type": [0, 2, 3, 4, 5, 6], "set_default_dtyp": 0, "neuralnetwork": 0, "modul": [0, 2, 3, 4, 5, 6], "__init__": [0, 1, 2, 3, 4, 5, 6], "self": [0, 1, 2, 3, 4, 5, 6], "n_hidden": 0, "20": [0, 1, 2, 3, 4, 5], "arg": 0, "int": [0, 2, 3, 4, 5, 6], "number": [0, 1, 3, 5], "super": [0, 1, 2, 3, 4, 5, 6], "linear": 0, "sequenti": [0, 3, 5], "one": 0, "forward": [0, 1, 2, 3, 4, 5, 6], "train_loop": 0, "dataload": [0, 2, 3, 4, 5, 6], "loss_fn": 0, "optim": [0, 1, 2, 3, 4, 5, 6], "mode": [0, 6], "num_batch": [0, 2, 3, 4, 5, 6], "train_loss": [0, 2, 3, 4, 5], "loss": [0, 1, 3, 6], "squeez": [0, 1], "backpropag": [0, 1], "updat": [0, 1], "zero_grad": [0, 1, 6], "zero": [0, 1, 2, 4, 6], "out": [0, 1], "previou": [0, 1, 6], "iter": [0, 1, 2, 6], "replac": [0, 1], "them": 0, "backward": [0, 1, 6], "step": [0, 1, 2, 4, 5, 6], "no_grad": [0, 1], "item": [0, 1, 6], "after": [0, 3, 5], "instal": [0, 1, 2, 3, 4, 5, 6], "save": 0, "tensor": [0, 1, 2, 3, 4, 5, 6], "util": [0, 2, 3, 4, 5, 6], "tensordataset": [0, 2, 3, 4, 5, 6], "turn": [0, 3, 5], "arrai": [0, 1, 2, 3, 4, 5, 6], "x_tensor": 0, "from_numpi": [0, 2, 3, 4, 5, 6], "column_stack": 0, "y_tensor": 0, "dataset": [0, 2, 3, 4, 5], "can": [0, 1, 2, 3, 4, 5, 6], "finish": 0, "when": 0, "desir": [0, 1], "epoch": [0, 2, 3, 4, 5], "ha": 0, "been": [0, 2], "reach": [0, 1, 2, 3, 4, 5], "also": [0, 1, 2], "some": [0, 2, 4, 5], "term": [0, 2], "pass": 0, "entir": [0, 3, 5], "rate": [0, 1], "determin": 0, "try": 0, "minim": 0, "faster": 0, "would": [0, 2], "have": [0, 1, 2], "larger": [0, 3, 5], "stochast": 0, "descent": 0, "sgd": [0, 1], "algorithm": 0, "note": [0, 3, 5], "exampl": [0, 2], "broken": 0, "87": 0, "batch": [0, 2, 3, 4, 5], "random": [0, 1, 4, 5], "seed": [0, 2, 5], "reproduc": [0, 4, 5, 6], "manual_se": 0, "314": 0, "hyperparamet": 0, "batch_siz": [0, 2, 3, 4, 5], "32": [0, 1, 2, 3, 4, 5, 6], "1000": 0, "learning_r": [0, 2, 3, 4, 5, 6], "1e": 0, "train_load": [0, 2, 3, 4, 5], "shuffl": [0, 1], "mseloss": 0, "lr": [0, 1, 2, 3, 4, 5, 6], "3d": [0, 1], "3f": [0, 1, 2, 3, 4, 5, 6], "done": [0, 1, 3, 5], "99": [0, 2, 4, 5], "242": 0, "199": 0, "019": [0, 6], "299": 0, "576": 0, "399": 0, "281": 0, "499": 0, "142": 0, "599": 0, "992": 0, "699": 0, "871": 0, "799": 0, "842": 0, "899": 0, "785": 0, "999": 0, "745": 0, "final": [0, 2, 3, 5, 6], "analyt": [0, 1], "between": [0, 1, 2, 3, 4, 5, 6], "eval": [0, 1, 6], "evalu": 0, "z_pred": [0, 1], "flatten": [0, 1], "detach": [0, 1, 2, 3, 4, 5, 6], "reshap": [0, 1, 2, 3, 4, 5], "shape": [0, 1, 2, 3, 4, 5, 6], "z_diff": 0, "order": 0, "closer": 0, "plug": 0, "name": [0, 1, 2, 3, 4, 5], "param": [0, 1, 2, 3, 4, 5], "named_paramet": [0, 1], "weights0": 0, "elif": [0, 2, 3, 4, 5], "bias0": 0, "weights2": 0, "bias2": 0, "xy0": 0, "z0": 0, "v1": 0, "i": [0, 1, 2, 3, 4, 5, 6], "v2": 0, "second": 0, "in_featur": [0, 2, 3, 4, 5], "out_featur": [0, 2, 3, 4, 5], "13": [0, 1], "0164": 0, "grad_fn": [0, 6], "viewbackward0": 0, "016363263303976": 0, "profession": 0, "infrom": 0, "logs_csv": [0, 2, 3, 4, 5], "directori": 0, "pip": [0, 1, 2, 3, 4, 5, 6], "quiet": 0, "pl": [0, 2, 3, 4, 5, 6], "logger": [0, 2, 3, 4, 5, 6], "csvlogger": [0, 2, 3, 4, 5], "you": [0, 1, 6], "mai": 0, "restart": 0, "kernel": [0, 6], "packag": [0, 1, 2, 6], "neuralnetworklightn": 0, "lightningmodul": [0, 2, 3, 4, 5], "training_step": [0, 2, 3, 4, 5], "batch_idx": [0, 2, 3, 4, 5], "y_pred": [0, 6], "mse_loss": [0, 2, 3, 4, 5], "log": [0, 1, 2, 3, 4, 5, 6], "configure_optim": [0, 2, 3, 4, 5], "adam": [0, 1, 2, 3, 4, 5, 6], "optimz": 0, "instead": [0, 2], "captur": [0, 1, 2, 3, 4, 5, 6], "seed_everyth": [0, 2, 4, 5], "worker": 0, "csv_logger": [0, 2, 3, 4, 5], "version": [0, 2, 3, 4, 5], "trainer": [0, 2, 3, 4, 5], "max_epoch": [0, 2, 3, 4, 5], "determinist": 0, "gpu": [0, 2, 3, 4, 5], "avail": [0, 2, 3, 4, 5], "fals": [0, 1, 2, 3, 4, 5], "tpu": [0, 2, 3, 4, 5], "core": [0, 2, 3, 4, 5], "ipu": [0, 2, 3, 4, 5], "hpu": [0, 2, 3, 4, 5], "81": 0, "trainabl": [0, 2, 3, 4, 5], "non": [0, 2, 3, 4, 5], "total": [0, 2, 3, 4, 5], "000": [0, 1], "estim": [0, 2, 3, 4, 5], "mb": [0, 2, 3, 4, 5], "stop": [0, 2, 3, 4, 5], "increas": [0, 1], "panda": [0, 2, 3, 4, 5], "pd": [0, 2, 3, 4, 5], "read_csv": [0, 2, 3, 4, 5], "lightning_log": [0, 2, 3, 4, 5], "version_0": 0, "metric": [0, 2, 3, 4, 5], "csv": [0, 2, 3, 4, 5], "subplot": [0, 1, 2, 3, 4, 5, 6], "semilog": [0, 2, 3, 4, 5], "set_xlabel": [0, 2, 3, 4, 5, 6], "set_ylabel": [0, 2, 3, 4, 5, 6], "text": [0, 3], "again": 0, "mueller_brown_potential_gradi": 0, "dx": [0, 1], "dy": [0, 1], "mueller_brown_potential_gradient_vector": 0, "dx_train": 0, "dy_train": 0, "dx_tensor": 0, "train_loop_with_gradi": 0, "train_loss_energi": 0, "train_loss_gradi": 0, "dx_pred": [0, 1], "autograd": [0, 2, 3, 4, 5, 6], "grad": [0, 2, 3, 4, 5, 6], "grad_output": 0, "ones_lik": 0, "create_graph": [0, 2, 3, 4, 5], "loss_energi": 0, "loss_gradi": 0, "requires_grad_": [0, 2, 3, 4, 5], "155": [0, 3], "851": [0, 6], "925": 0, "144": 0, "926": 0, "487": 0, "643": 0, "70": [0, 1], "844": 0, "31": [0, 2], "353": 0, "031": 0, "30": [0, 1], "322": 0, "27": [0, 2, 4, 5, 6], "340": 0, "847": 0, "26": 0, "493": 0, "25": [0, 3, 5], "616": 0, "818": 0, "24": 0, "797": [0, 1], "661": 0, "823": 0, "23": [0, 2, 4, 5, 6], "838": 0, "933": 0, "089": 0, "246": 0, "827": 0, "22": 0, "418": 0, "167": 0, "756": 0, "21": [0, 3, 4, 5], "412": [0, 1], "067": 0, "692": 0, "375": 0, "start": 1, "function": [1, 3, 5, 7], "deriv": 1, "respect": [1, 3], "muller": 1, "frac": [1, 2, 3, 5], "dv": 1, "2a_k": 1, "A": [1, 2, 5], "dx_valu": 1, "dy_valu": 1, "neural": [1, 7], "network": [1, 7], "previous": [1, 2, 3, 4, 5, 6], "lesson": [1, 4, 7], "displai": [1, 2, 4, 5], "test": [1, 2, 4, 5, 6], "xx": 1, "yy": 1, "xy": [1, 2, 4, 5, 6], "xy_trunc": 1, "z_truncat": 1, "dx_truncat": 1, "dy_trunc": 1, "dvx": 1, "dvy": 1, "append": [1, 2, 3, 4, 5], "ml": [1, 4], "zmin": 1, "amin": 1, "zmax": 1, "amax": 1, "x_truncat": 1, "y_truncat": 1, "x_ref": 1, "y_ref": 1, "z_ref": 1, "599803525171698": 1, "4772333054245": 1, "896": 1, "To": 1, "more": 1, "readabl": 1, "extrem": 1, "high": 1, "nan": 1, "same": 1, "help": 1, "ignor": 1, "region": 1, "interest": 1, "becaus": [1, 2], "accur": [1, 2], "reflect": 1, "copi": 1, "clean_z": 1, "allow": [1, 2], "cm": 1, "m\u00fceller": 1, "let": 1, "variabl": [1, 6], "featur": 1, "observ": [1, 2, 6], "textbf": 1, "x_1": 1, "x_d": 1, "n": [1, 3, 5], "configur": 1, "assembl": 1, "_1": [1, 3, 5], "_n": 1, "correspond": 1, "y_1": 1, "y_n": 1, "t": [1, 2, 3, 4, 5, 6], "noisi": 1, "sampl": [1, 4, 5], "assum": 1, "seper": 1, "underli": 1, "accord": 1, "mathit": 1, "\u03b5": 1, "where": [1, 2, 3, 5], "nois": [1, 6], "follow": [1, 2, 4, 5], "distribut": [1, 2, 6], "sim": 1, "mathcal": [1, 3, 5], "\u03c3": 1, "2_n": 1, "sigma": 1, "paramet": [1, 2, 3, 4, 5, 6], "prior": 1, "mean": [1, 2, 3, 4, 5, 6], "covari": 1, "matrix": [1, 6], "base": 1, "measur": 1, "simular": 1, "begin": [1, 2, 3, 4, 5, 6], "bmatrix": 1, "ldot": 1, "vdot": 1, "ddot": 1, "end": [1, 2, 3, 5], "radial": [1, 2, 4, 6], "basi": 1, "_a": 1, "_b": 1, "2_f": 1, "2l": 1, "vertic": 1, "variat": 1, "l": 1, "length": [1, 2], "setup": 1, "taken": 1, "directli": 1, "minor": 1, "chang": [1, 2], "class": 1, "exactgpmodel": 1, "exactgp": [1, 6], "train_x": 1, "train_i": 1, "likelihood": [1, 6], "mean_modul": [1, 6], "zeromean": 1, "covar_modul": [1, 6], "scalekernel": [1, 6], "rbfkernel": [1, 6], "mean_x": [1, 6], "covar_x": [1, 6], "multivariatenorm": [1, 6], "x_gpr": 1, "z_gpr": 1, "initi": 1, "gaussianlikelihood": [1, 6], "With": [1, 4, 5, 6], "tild": 1, "being": 1, "ident": 1, "theta": 1, "maxim": 1, "margin": [1, 6], "p": 1, "\u03b8": 1, "pi": [1, 2, 3, 4, 5, 6], "demonstr": 1, "neg": 1, "smooth": [1, 7], "hold": 1, "constant": 1, "vari": 1, "noise_valu": 1, "list": 1, "scale_and_length": 1, "j": [1, 2, 3, 5], "50": [1, 3, 5], "x_plt": 1, "y_plt": 1, "z_plt": 1, "pair": 1, "hyper": 1, "noise_covar": 1, "base_kernel": [1, 6], "lengthscal": [1, 6], "outputscal": [1, 6], "calcul": [1, 2, 4, 5, 6], "mll": [1, 6], "exactmarginalloglikelihood": [1, 6], "call": [1, 2, 4, 6], "tricontour": 1, "tricontourf": 1, "set_label": 1, "rotat": [1, 2, 3, 4, 5], "270": 1, "built": 1, "until": 1, "Then": [1, 6], "train_model": 1, "print_hp": 1, "untrain": 1, "param_nam": 1, "42": 1, "5f": 1, "training_it": [1, 6], "find": [1, 6], "gp": [1, 6], "raw_nois": 1, "54117": 1, "raw_outputscal": 1, "54132": 1, "raw_lengthscal": 1, "879": 1, "263": 1, "046": 1, "787": 1, "136": 1, "360": 1, "087": 1, "633": 1, "209": 1, "40": 1, "854": 1, "119": 1, "536": 1, "250": 1, "479": 1, "149": 1, "459": 1, "272": 1, "60": 1, "255": 1, "178": 1, "405": 1, "279": 1, "135": 1, "204": 1, "369": 1, "80": 1, "063": 1, "230": 1, "345": 1, "275": 1, "90": 1, "014": 1, "254": 1, "327": 1, "268": 1, "980": [1, 6], "313": 1, "260": 1, "91370": 1, "98492": 1, "03970": 1, "new": 1, "space": [1, 2], "\u03bc": 1, "abov": 1, "equaiton": 1, "show_surfac": 1, "z_var": 1, "zdiff": 1, "subtract": 1, "47233": 1, "96658": 1, "08929": 1, "30226": 1, "recal": 1, "compos": 1, "depend": [1, 3, 5], "upon": 1, "well": 1, "partial": 1, "themselv": 1, "consequ": 1, "thei": 1, "incorpor": 1, "target": 1, "explicit": 1, "_": [1, 2, 3, 4, 5, 6], "ext": 1, "account": 1, "addit": 1, "displaystyl": 1, "gpmodelwithderiv": 1, "constantmeangrad": 1, "rbfkernelgrad": 1, "ard_num_dim": 1, "multitaskmultivariatenorm": 1, "dx_gpr": 1, "dy_gpr": 1, "y_gpr": 1, "stack": 1, "includ": [1, 6], "x_test": [1, 6], "y_test": 1, "multitaskgaussianlikelihood": 1, "num_task": 1, "use_gpu": 1, "cuda": [1, 2, 3, 4, 5, 6], "is_avail": [1, 2, 3, 4, 5, 6], "28": 1, "textrm": [1, 2], "As": 1, "which": [1, 2, 3, 4, 5, 6], "contain": [1, 2, 4, 5], "m": 1, "19": 1, "531": 1, "693": 1, "611": 1, "744": 1, "644": 1, "798": 1, "598": 1, "883": 1, "852": 1, "556": 1, "555": 1, "850": 1, "830": 1, "908": 1, "516": 1, "515": 1, "903": 1, "109": 1, "965": 1, "480": 1, "478": 1, "955": 1, "740": 1, "021": 1, "446": 1, "444": 1, "003": 1, "752": 1, "078": 1, "416": 1, "049": 1, "085": 1, "133": 1, "388": 1, "383": [1, 2, 4], "090": 1, "655": 1, "188": 1, "364": 1, "358": 1, "128": 1, "its": 1, "expect": 1, "e": [1, 2, 3], "2_": 1, "ast_": 1, "ast": 1, "associ": 1, "var": 1, "q": [1, 6], "bigg": 1, "fast_comput": 1, "log_prob": 1, "covar_root_decomposit": 1, "cpu": [1, 2, 3, 4, 5, 6], "n1": 1, "n2": 1, "dx_diff": 1, "dy_pr": 1, "dy_diff": 1, "work": 1, "look": 1, "accuraci": [1, 5], "root": 1, "squar": 1, "error": [1, 2], "rmse": 1, "decreas": 1, "r": [1, 2, 3, 4, 5, 6], "tabul": 1, "z_test": 1, "evaluate_model": 1, "train_z": 1, "test_x": 1, "test_z": 1, "preds_train": 1, "preds_test": 1, "rmse_train": 1, "sqrt": [1, 2, 3, 4, 5, 6], "rmse_test": 1, "r2": 1, "sum": [1, 2, 3, 4, 5, 6], "q2": [1, 6], "reduce_training_set": 1, "new_siz": 1, "arr_index": 1, "new_model": 1, "size_list": 1, "rmse_train_list": 1, "r2_list": 1, "rmse_test_list": 1, "q2_list": 1, "training_set_s": 1, "600": 1, "400": [1, 5, 6], "300": [1, 2, 4, 5, 6], "set_siz": 1, "training_set_dict": 1, "header": 1, "kei": 1, "floatfmt": 1, "4f": [1, 6], "opt": 1, "hostedtoolcach": 1, "x64": 1, "lib": [1, 6], "python3": [1, 6], "site": 1, "exact_gp": [1, 6], "py": [1, 6], "284": [1, 6], "gpinputwarn": [1, 6], "match": [1, 6], "store": [1, 6], "did": [1, 6], "forget": [1, 6], "0466": 1, "9756": 1, "0515": 1, "9731": 1, "0679": 1, "0592": 1, "9689": 1, "0636": 1, "0660": 1, "9592": 1, "0726": 1, "0846": 1, "9415": 1, "0257": 1, "1065": 1, "9190": 1, "0351": 1, "1125": 1, "8442": 1, "0584": 1, "result": [1, 2], "befor": 1, "rbf": 1, "matern": 1, "ration": 1, "quadrat": 1, "rq": 1, "gpmodel_kernel": 1, "constantmean": [1, 6], "maternkernel": 1, "rqkernel": 1, "train_model_k": 1, "calc": [1, 6], "backprop": [1, 6], "kernel_nam": 1, "str": [1, 2, 3, 4, 5], "split": 1, "run": 1, "0031": 1, "0026": 1, "0021": 1, "atom": [2, 3, 4, 5, 6], "center": 2, "ensur": [2, 3, 4, 5], "invari": [2, 4], "translat": [2, 3, 4, 5], "requir": 2, "lightn": 2, "dev": [2, 3, 4, 5, 6], "null": [2, 3, 4, 5, 6], "sequenc": [2, 3, 4, 5, 6], "tupl": [2, 3, 4, 5, 6], "nn": [2, 3, 4, 5, 6], "random_split": [2, 3, 4, 5, 6], "pytorch_lightn": [2, 3, 4, 5, 6], "pl_logger": [2, 3, 4, 5, 6], "cartesian": 2, "coordin": [2, 3, 4, 5, 6], "brown": 2, "potenti": 2, "machin": [2, 3], "howev": 2, "ideal": 2, "molecul": [2, 3, 5], "case": [2, 3, 5], "chemistri": 2, "impli": 2, "incorrect": 2, "describ": [2, 3, 4, 5], "newli": 2, "transform": [2, 3, 5], "cutoff": [2, 3, 5], "sever": 2, "were": 2, "introduc": 2, "2007": 2, "http": [2, 3, 4, 5, 6], "doi": [2, 3], "org": [2, 3], "1103": 2, "physrevlett": 2, "98": 2, "146401": 2, "align": 2, "f_c": 2, "r_": [2, 3, 5], "ij": 2, "cl": 2, "time": [2, 3, 4, 5], "co": [2, 3, 4, 5, 6], "r_c": [2, 3, 5], "le": 2, "gt": 2, "distanc": [2, 3, 4, 5, 6], "g_i": 2, "neq": 2, "all": 2, "eta": 2, "gaussian": [2, 7], "shift": 2, "peak": 2, "angular": [2, 4, 6], "zeta": [2, 4, 6], "lambda": 2, "theta_": 2, "ijk": [2, 4, 6], "ik": 2, "jk": 2, "3": [2, 3, 4, 5, 6], "mathbf": 2, "cdot": [2, 3, 5], "give": [2, 3, 5], "explor": 2, "environ": 2, "posit": 2, "2017": 2, "smith": 2, "isayev": 2, "roitberg": 2, "develop": 2, "anakin": 2, "me": 2, "engin": 2, "molecular": 2, "refer": [2, 3, 4, 5, 6], "1039": 2, "c6sc05720a": 2, "modifi": 2, "shell": 2, "probe": 2, "differ": 2, "angl": [2, 4, 6], "pairwise_vector": [2, 4, 6], "symmetry_function_g1": [2, 4, 6], "bp": [2, 7], "symmetry_function_g2": [2, 4, 6], "symmetry_function_g2ani": [2, 4, 6], "coord": [2, 3, 4, 5, 6], "num_channel": [2, 3, 4, 5, 6], "rij": [2, 3, 4, 5, 6], "none": [2, 3, 4, 5, 6], "mask": [2, 3, 4, 5, 6], "ey": [2, 3, 4, 5, 6], "bool": [2, 3, 4, 5, 6], "devic": [2, 3, 4, 5, 6], "remov": [2, 3, 4, 5, 6], "masked_select": [2, 3, 4, 5, 6], "unsqueez": [2, 3, 4, 5, 6], "view": [2, 3, 4, 5, 6], "rcr": [2, 4, 6], "float": [2, 4, 6], "etar": [2, 4, 6], "shfr": [2, 4, 6], "dij": [2, 3, 4, 5, 6], "norm": [2, 3, 4, 5, 6], "dim": [2, 3, 4, 5, 6], "fij": [2, 4, 6], "g1": [2, 4, 6], "rca": [2, 4, 6], "etaa": [2, 4, 6], "lama": [2, 4, 6], "combin": [2, 4, 5, 6], "r12": [2, 4, 6], "r13": [2, 4, 6], "r23": [2, 4, 6], "d12": [2, 4, 6], "d13": [2, 4, 6], "d23": [2, 4, 6], "f12": [2, 4, 6], "f13": [2, 4, 6], "f23": [2, 4, 6], "cosin": [2, 4, 6], "einsum": [2, 4, 6], "ijkl": [2, 4, 6], "g2": [2, 4, 6], "shfz": [2, 4, 6], "shfa": [2, 4, 6], "aco": [2, 4, 6], "feed": [2, 4], "aspect": [2, 4], "connect": [2, 4, 5], "residu": [2, 3, 4, 5], "els": [2, 3, 4, 5, 6], "register_paramet": [2, 3, 4, 5], "reset_paramet": [2, 3, 4, 5], "init": [2, 3, 4, 5], "kaiming_uniform_": [2, 3, 4, 5], "zip": [2, 3, 4, 5], "fan_in": [2, 3, 4, 5], "_calculate_fan_in_and_fan_out": [2, 3, 4, 5], "bound": [2, 3, 4, 5], "uniform_": [2, 3, 4, 5], "channel": [2, 3, 4, 5], "bmm": [2, 3, 4, 5], "transpos": [2, 3, 4, 5], "cat": [2, 3, 4, 5], "rais": [2, 3, 4, 5], "notimplementederror": [2, 3, 4, 5], "Not": [2, 3, 4, 5], "implement": [2, 4, 5, 7], "extra_repr": [2, 3, 4, 5], "format": [2, 3, 4, 5], "take": [2, 4, 6], "inform": [2, 4, 6], "characterist": [2, 4, 6], "uniqu": [2, 3, 4, 5, 6], "assert": [2, 4, 6], "atom_typ": [2, 3, 4, 5, 6], "concat": [2, 4, 6], "properti": [2, 3, 4, 5, 6], "output_length": [2, 3, 4, 5, 6], "featureani": [2, 4, 6], "n_type": [2, 3, 4, 5], "240": [2, 3, 4, 5], "fitting_net": [2, 3, 4, 5], "descriptor": [2, 3, 4, 5, 6], "5e": [2, 3, 4, 5, 6], "atomic_energi": [2, 3, 4, 5], "unbind": [2, 3, 4, 5], "gradient": [2, 3, 4, 5, 6], "hstack": [2, 3, 4, 5], "qm_coord": [2, 3, 4, 5, 6], "ene_pr": [2, 3, 4, 5], "grad_pr": [2, 3, 4, 5], "ene_loss": [2, 4, 5], "grad_loss": [2, 4, 5], "param_group": [2, 4, 5], "start_lr": [2, 4, 5], "initial_lr": [2, 4, 5], "w_ene": [2, 4, 5], "w_grad": [2, 4, 5], "l2_trn": [2, 4, 5], "l2_e_trn": [2, 4, 5], "l2_f_trn": [2, 4, 5], "validation_step": [2, 4, 5], "set_grad_en": [2, 4, 5], "val_loss": [2, 4, 5], "l2_tst": [2, 4, 5], "l2_e_tst": [2, 4, 5], "l2_f_tst": [2, 4, 5], "schedul": [2, 3, 4, 5], "lr_schedul": [2, 3, 4, 5], "exponentiallr": [2, 3, 4, 5], "95": [2, 3, 4, 5, 6], "interv": [2, 3, 4, 5], "frequenc": [2, 3, 4, 5], "github": [2, 3, 4, 5, 6], "npy": [2, 3, 4, 5, 6], "1800": 2, "qm_elem": [2, 4, 5, 6], "txt": [2, 4, 5, 6], "pm3": [2, 4, 5, 6], "energy_sqm": [2, 4, 5, 6], "qm_grad_sqm": [2, 4, 5, 6], "b3lyp": [2, 4, 5, 6], "g": [2, 3, 5], "qm_grad": [2, 4, 5, 6], "These": 2, "provid": 2, "semi": 2, "empir": 2, "qm": 2, "ds": [2, 3, 4, 5, 6], "datasourc": [2, 3, 4, 5, 6], "open": [2, 3, 4, 5, 6], "com": [2, 3, 4, 5, 6], "cc": [2, 3, 4, 5, 6], "ats": [2, 3, 4, 5, 6], "mlp_tutori": [2, 3, 4, 5, 6], "raw": [2, 3, 4, 5, 6], "main": [2, 3, 4, 5, 6], "butan": 2, "rb": [2, 3, 4, 5, 6], "loadtxt": [2, 3, 4, 5, 6], "elem": [2, 3, 4, 5, 6], "tolist": [2, 3, 4, 5, 6], "index": [2, 3, 4, 5, 6], "repeat": [2, 3, 4, 5, 6], "2114": [2, 4, 5, 6], "061": [2, 4, 5, 6], "qm_gradient": [2, 4, 5, 6], "529177249": [2, 4, 5, 6], "val": [2, 4, 5], "1728": 2, "72": 2, "96": [2, 4, 5], "1780": 2, "val_load": [2, 4, 5], "info": [2, 3, 4, 5], "lightning_fabr": [2, 4, 5], "global": [2, 4, 5], "ase_ani": 2, "serv": 2, "interfac": 2, "torchani": [2, 4, 6], "1021": 2, "ac": 2, "jcim": 2, "0c00451": 2, "softwar": 2, "2000e": [2, 4, 6], "00": [2, 4, 6], "5000e": [2, 4, 6], "6000000e": [2, 4, 6], "01": [2, 4, 6], "0000000e": [2, 4, 6], "1687500e": [2, 4, 6], "4375000e": [2, 4, 6], "7062500e": [2, 4, 6], "9750000e": [2, 4, 6], "2437500e": [2, 4, 6], "5125000e": [2, 4, 6], "7812500e": [2, 4, 6], "0500000e": [2, 4, 6], "3187500e": [2, 4, 6], "5875000e": [2, 4, 6], "8562500e": [2, 4, 6], "1250000e": [2, 4, 6], "3937500e": [2, 4, 6], "6625000e": [2, 4, 6], "9312500e": [2, 4, 6], "2000000e": [2, 4, 6], "9634954e": [2, 4, 6], "8904862e": [2, 4, 6], "8174770e": [2, 4, 6], "3744468e": [2, 4, 6], "7671459e": [2, 4, 6], "1598449e": [2, 4, 6], "5525440e": [2, 4, 6], "9452431e": [2, 4, 6], "5500000e": [2, 4, 6], "8500000e": [2, 4, 6], "everi": [2, 4, 5], "h": [2, 4, 6], "o": [2, 4, 6], "04": [2, 4, 6], "71": [2, 4, 6], "79": [2, 4, 6], "16": [2, 4, 6], "log_every_n_step": [2, 4, 5], "acceler": [2, 3, 4, 5], "auto": [2, 3, 4, 5], "rank_zero": [2, 3, 4, 5], "local_rank": [2, 3, 4, 5], "cuda_visible_devic": [2, 3, 4, 5], "callback": [2, 3, 4, 5], "model_summari": [2, 3, 4, 5], "532": [2, 4], "pt": [2, 4, 5], "model_script": [2, 4, 5], "jit": [2, 4, 5], "attempt": [2, 4, 5], "preserv": [2, 3, 4, 5], "behavior": [2, 4, 5], "oper": [2, 4, 5], "across": [2, 4, 5], "load_state_dict": [2, 4, 5], "state_dict": [2, 4, 5], "to_torchscript": [2, 4, 5], "e1": [2, 4, 5, 6], "e2": [2, 4, 5, 6], "linestyl": [2, 3, 4, 5, 6], "marker": [2, 3, 4, 5, 6], "mediumspringgreen": 2, "linewidth": [2, 3, 4, 5, 6], "kcal": [2, 3, 4, 5, 6], "mol": [2, 3, 4, 5, 6], "annot": [2, 4, 5, 6], "05": [2, 4, 5, 6], "xycoord": [2, 4, 5, 6], "fraction": [2, 4, 5, 6], "f1": [2, 3, 4, 5, 6], "f2": [2, 3, 4, 5, 6], "aa": [2, 3, 4, 5, 6], "savefig": [2, 4, 5, 6], "png": [2, 4, 5, 6], "cell": [2, 4, 5], "param_tensor": [2, 4, 5], "33": 2, "48": [2, 4], "version_": [2, 3, 4, 5], "isnul": [2, 4, 5], "y2": [2, 4], "dodgerblu": [2, 3, 4, 5], "set_titl": 2, "label": [2, 4, 5], "xtick": 2, "ytick": 2, "process": [2, 4, 5, 7], "legend": [2, 4, 5], "1350": 2, "xiaoliang": 3, "pan": 3, "2022": 3, "origin": 3, "paper": 3, "symmetri": [3, 5, 7], "inter": 3, "energi": 3, "model": 3, "finit": 3, "system": 3, "linfeng": 3, "zhang": 3, "jiequn": 3, "han": 3, "wang": 3, "wissam": 3, "saidi": 3, "roberto": 3, "car": 3, "weinan": 3, "48550": 3, "arxiv": 3, "1805": 3, "09003": 3, "build": [3, 5], "envrion": [3, 5], "d_i": [3, 5], "map": [3, 5], "matric": [3, 5], "manner": [3, 5], "permut": [3, 4, 5], "obtain": [3, 4, 5, 6], "togeth": [3, 5], "extens": [3, 5], "shown": [3, 4, 5], "_i": [3, 5], "eqnarrai": [3, 5], "pmatrix": [3, 5], "s": [3, 5], "1i": [3, 5], "x_": [3, 5], "y_": [3, 5], "z_": [3, 5], "2i": [3, 5], "i1": [3, 5], "ni": [3, 5], "neighbor": [3, 5], "ji": [3, 5], "cs": [3, 5], "down": [3, 5], "becom": [3, 5], "greater": [3, 5], "than": [3, 5], "approach": [3, 5], "It": [3, 5], "beyond": [3, 5], "continu": [3, 5], "differenti": [3, 5], "_environ": [3, 5], "local_environ": [3, 5], "dij_inv": [3, 5], "dij2_inv": [3, 5], "loc_env_r": [3, 5], "loc_env_a": [3, 5], "gener": [3, 4, 5], "compon": [3, 5], "while": [3, 4, 5], "scheme": [3, 5], "present": [3, 5], "m_1": [3, 5], "chemic": [3, 5], "speci": [3, 5], "_2": [3, 5], "two": [3, 5], "dimens": [3, 5], "i2": [3, 5], "m_2": [3, 5], "repres": [3, 5], "By": [3, 5], "multipli": [3, 5], "form": [3, 5], "encod": [3, 5], "d": [3, 5, 6], "yield": [3, 5], "1_i": [3, 5], "2_i": [3, 5], "construct": [3, 4, 5], "axis_neuron": [3, 5], "local_embed": [3, 5], "neighbor_typ": [3, 5], "indic": [3, 5], "load": [3, 4, 5, 6], "deeppot_pytorch": 3, "input_coord": 3, "input_grad": 3, "120": 3, "176": 3, "707": 3, "springgreen": [3, 4, 5, 6], "set_aspect": 3, "equal": 3, "adjust": 3, "box": 3, "18": 3, "reduct": 3, "bpnn": 4, "delta": [4, 5, 6], "reaction": [4, 5, 6], "goal": [4, 5, 6], "semiempir": [4, 5, 6], "dft": [4, 5, 6], "theori": [4, 5, 6], "correct": [4, 5, 6], "stretch": [4, 5], "d_2": [4, 5], "bond": [4, 5], "shrink": [4, 5], "d_1": [4, 5], "window": [4, 5], "along": [4, 5], "2100": [4, 5], "frame": [4, 5], "ps": [4, 5], "fs": [4, 5], "ipython": [4, 5], "html": [4, 5], "video": [4, 5], "src": [4, 5], "claisen_rearrang": [4, 5, 6], "img": [4, 5], "mp4": [4, 5], "control": [4, 5], "predict": [4, 5], "2016": [4, 5], "84": [4, 5], "315": 4, "0x7cc48e7a5870": 4, "claisen": [5, 6, 7], "rearrang": [5, 6, 7], "58": 5, "636": 5, "694": 5, "778": 5, "945": 5, "mlp": 6, "gpytorch": 6, "bpgpr": 6, "nskip": 6, "701": 6, "948": 6, "301": 6, "517": 6, "565": 6, "562": 6, "193": 6, "451": 6, "942": 6, "755": 6, "075": 6, "395": 6, "174": 6, "034": 6, "392": 6, "245": 6, "153": 6, "255328178405762": 6, "01911887526512146": 6, "prepar": 6, "requires_grad": 6, "y_mean": 6, "y_var": 6, "varianc": 6, "y_covar": 6, "covariance_matrix": 6, "ref": 6, "perform": 6, "y_rmse": 6, "y_q2": 6, "usr": 6, "local": 6, "dist": 6, "warn": 6, "0752": 6, "meanbackward0": 6, "04933705925941467": 6, "9998255372047424": 6, "auto_grad": 6, "0737624168396": 6, "8090568780899048": 6, "concaten": 6, "ab": 6, "qm_coord_train": 6, "0493": 6, "9998": 6, "0738": 6, "8091": 6, "fit": 7, "regress": 7, "behler": 7, "parrinello": 7, "pytorch": 7, "deep": 7, "edit": 7, "deeppot": 7, "se": 7, "fnn": 7, "gpr": 7}, "objects": {}, "objtypes": {}, "objnames": {}, "titleterms": {"fit": [0, 2, 3, 4, 5], "neural": [0, 2, 3, 4, 5], "network": [0, 2, 3, 4, 5], "model": [0, 1, 2, 4, 5, 6, 7], "defin": [0, 1, 2, 3, 4, 5, 6], "m\u00fcller": 0, "brown": [0, 1], "potenti": [0, 1, 3, 4, 5, 6, 7], "energi": [0, 1, 2, 4, 5, 6], "function": [0, 2, 4, 6], "gener": [0, 1], "train": [0, 1, 2, 3, 4, 5, 6], "data": [0, 1, 2, 3, 4, 5, 6], "visual": [0, 1], "3": [0, 1], "d": [0, 1], "project": [0, 1], "surfac": [0, 1], "contour": [0, 1], "class": [0, 2, 3, 4, 5, 6], "A": [0, 3, 7], "basic": 0, "load": [0, 1, 2], "pytorch": [0, 1, 2, 3, 4, 5, 6], "plot": [0, 1, 2, 3, 4, 5, 6], "refer": [0, 1], "predict": [0, 1, 2, 3, 6], "differ": [0, 1], "take": 0, "look": 0, "nn": 0, "paramet": 0, "more": 0, "autom": 0, "refin": 0, "implement": [0, 3], "lightn": [0, 3, 4, 5, 6], "error": [0, 3, 4, 5], "gradient": [0, 1], "gaussian": [1, 6], "process": [1, 6], "regress": [1, 6], "mueller": 1, "gpytorch": 1, "gpr": [1, 6], "learn": [1, 4, 5, 6, 7], "hyperparamet": [1, 6], "varianc": 1, "us": [1, 2, 4, 6], "perform": 1, "set": 1, "size": 1, "compar": 1, "kernel": 1, "behler": [2, 4, 6], "parrinello": [2, 4, 6], "symmetri": [2, 4, 6], "import": [2, 3, 4, 5, 6], "librari": [2, 3, 4, 5, 6], "featur": [2, 3, 4, 5, 6], "extract": [2, 3, 4, 5, 6], "sequenti": [2, 4], "dens": [2, 3, 4, 5], "creat": [2, 4, 6], "ani": [2, 4, 6], "bpnn": 2, "save": [2, 4, 5], "file": [2, 4, 5], "evalu": [2, 4, 6], "s": [2, 4, 6], "accuraci": [2, 4, 6], "rmsd": [2, 3, 4, 5, 6], "forc": [2, 3, 4, 5, 6], "The": [2, 3, 4, 5], "weight": [2, 4, 5], "bias": [2, 4, 5], "dictionari": [2, 4, 5], "minim": [2, 4], "loss": [2, 4, 5], "valid": [2, 4, 5], "deep": 3, "smooth": [3, 5], "edit": [3, 5], "deeppot": [3, 5], "se": [3, 5], "local": [3, 5], "environ": [3, 5], "embed": [3, 5], "matrix": [3, 5], "machin": [4, 5, 6, 7], "bp": [4, 6], "fnn": [4, 5], "mlp": [4, 5, 7], "claisen": 4, "rearrang": 4, "seed": [4, 6], "deltamlp": [4, 5], "lesson": [5, 6], "6": 5, "7": 6, "reshap": 6, "initi": 6, "rmse": 6, "tabul": 6, "tutori": 7, "tabl": 7, "content": 7, "b": 7, "molecular": 7, "represent": 7, "c": 7}, "envversion": {"sphinx.domains.c": 2, "sphinx.domains.changeset": 1, "sphinx.domains.citation": 1, "sphinx.domains.cpp": 6, "sphinx.domains.index": 1, "sphinx.domains.javascript": 2, "sphinx.domains.math": 2, "sphinx.domains.python": 3, "sphinx.domains.rst": 2, "sphinx.domains.std": 2, "sphinx.ext.intersphinx": 1, "sphinxcontrib.bibtex": 9, "sphinx": 56}})