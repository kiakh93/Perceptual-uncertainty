from tqdm import tqdm
from train_DEns import *

"""
Make sure to first run "extract_pu.py" to get the raw feature data.
"""

def main(DEns_number=50,
         run_number=1):

    root = 'Datasets/SRS/data from same section/test/Raw Features'
    f_name = os.listdir(root)

    # In order: Var, MAD, Binary Act. H, Multi Act. H
    pac_list = [[], [], [], []]
    pui_list = [[], [], [], []]
    PAvPU_list = [[], [], [], []]

    for n in range(run_number):
        Model_Index = random.sample(list(range(50)), DEns_number)

        uq_Var = []
        uq_MAD = []
        uq_BinActH = []
        uq_MultiActH = []

        error = []
        for f in tqdm(f_name):

            features = torch.load(root + '/' + f)[Model_Index,:] # (M, 2048, 16, 16)
            pred = features.mean(0)
            gt = torch.load('Datasets/SRS/data from same section/test/Raw Features (Ground truth)/'+ f)

            # Error
            err = ((gt - pred) ** 2).mean(0)
            error += [err]

            # Method 1: Var.
            f_var = torch.std(features, dim=0).square()
            var = f_var.mean(0)
            uq_Var += [var]

            # Method 2: MAD-based Var.
            f_median = torch.median(features, dim=0, keepdim=True)[0]
            f_MAD = (torch.median(torch.abs(features - f_median), dim=0)[0] * 1.4826).square()
            mad = f_MAD.mean(0)
            uq_MAD += [mad]

            # Method 3: Binary Act. H
            f_act = features.detach().clone()
            f_act[f_act > 0] = 1
            f_sfmx = f_act.mean(0)  # (2048, 16, 16)
            f_sfmx = torch.clamp(f_sfmx, min=1e-3)
            bah = (torch.special.entr(f_sfmx) + torch.special.entr(1.-f_sfmx)).mean(0)
            uq_BinActH += [bah]

            # Method 4: Multi Act. H
            f_int = torch.ceil(features)
            f_int[f_int > 9] = 9
            f_int2 = torch.empty((10, 2048, f_int.size(-1), f_int.size(-1))).to(f_int.device)
            for c in range(10):
                f_int2[c, :] = (f_int == c).sum(0) / DEns_number
            f_int2 = torch.clamp(f_int2, min=1e-3)
            mah = torch.special.entr(f_int2).sum(0).mean(0)
            uq_MultiActH += [mah]

        Error = torch.stack(error, dim=0)
        UQVar = torch.stack(uq_Var, dim=0)
        UQMAD = torch.stack(uq_MAD, dim=0)
        UQBAH = torch.stack(uq_BinActH, dim=0)
        UQMAH = torch.stack(uq_MultiActH, dim=0)

        # find accurate/inaccurate crossover points:
        err_q_list = []
        for q in range(10):
            err_q_list += [np.quantile(Error.detach().cpu().numpy(), q=0.90 + q / 100)]

        patch_size = Error.numel()

        for method, UQ in enumerate([UQVar, UQMAD, UQBAH, UQMAH]):

            uq = UQ.detach().cpu().numpy()

            # find certain/uncertain crossover points:
            unc_q_list = []
            for q in range(1, 11):
                unc_q_list += [np.quantile(uq, q=q / 10)]

            p_ac = []
            p_ui = []
            pavpu = []

            for error_threshold in err_q_list:

                c_list, ac_list, ui_list, pavpu_list = [0.], [1.], [1.], [1.]
                for unc_threshold in unc_q_list:
                    conf_num = max((UQ < unc_threshold).sum().item(), 1e-6)
                    inacc_num = max((Error >= error_threshold).sum().item(), 1e-6)

                    c_list += [conf_num / patch_size]
                    ac_list += [((UQ < unc_threshold) * (Error < error_threshold)).sum().item() / conf_num]
                    ui_list += [((UQ >= unc_threshold) * (Error >= error_threshold)).sum().item() / inacc_num]
                    pavpu_list += [
                        (((UQ < unc_threshold) * (Error < error_threshold)).sum().item() + (
                                    (UQ >= unc_threshold) * (Error >= error_threshold)).sum().item())
                        / (patch_size)
                    ]

                # append AUCs.
                p_ac += [np.trapz(ac_list, x=c_list)]
                p_ui += [np.trapz(ui_list, x=c_list)]
                pavpu += [np.trapz(pavpu_list, x=c_list)]

            # get mean AUCs over the different error cutoff points.
            pac_list[method] += [stats.mean(p_ac)]
            pui_list[method] += [stats.mean(p_ui)]
            PAvPU_list[method] += [stats.mean(pavpu)]

    for j in range(4):
        if run_number > 1:
            print(stats.mean(pac_list[j]), stats.stdev(pac_list[j]))
            print(stats.mean(pui_list[j]), stats.stdev(pui_list[j]))
            print(stats.mean(PAvPU_list[j]), stats.stdev(PAvPU_list[j]))
            print('\n')
        else:
            print(stats.mean(pac_list[j]))
            print(stats.mean(pui_list[j]))
            print(stats.mean(PAvPU_list[j]))
            print('\n')


if __name__ == '__main__':
    main()