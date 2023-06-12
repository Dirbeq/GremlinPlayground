from IsolationForest.IF import main as IFmain
from LocalOutlierFactor.LOF import main as LOFmain
from OneClassSVM.OCSvm import main as OCSVMmain
from Other.COPOD import main as COPODmain
from Other.GMM import main as GMMmain
from Other.INNE import main as INNEmain


def main():
    IFmain()
    LOFmain()
    OCSVMmain()
    INNEmain()
    GMMmain()
    COPODmain()


if __name__ == '__main__':
    main()
