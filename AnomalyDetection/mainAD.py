from IsolationForest.IF import main as IFmain
from LocalOutlierFactor.LOF import main as LOFmain
from LocalOutlierProbability.LOP import main as LOPmain
from OneClassSVM.OCSvm import main as OCSVMmain


def main():
    IFmain()
    LOFmain()
    LOPmain()
    OCSVMmain()


if __name__ == '__main__':
    main()
