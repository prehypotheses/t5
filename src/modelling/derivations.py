"""Module derivations.py"""
import pandas as pd


class Derivations:
    """
    <b>Additional Notes</b><br>
    -----------------<br>
    <b>False Negative Rate</b>: Type II Error, Miss Rate, 1 - Sensitivity<br><br>
    <b>False Positive Rate</b>: Type I Error, Probability of False Alarm, Fall Out Measure, 1 - Specificity<br>
    """

    def __init__(self, cases: pd.DataFrame):
        """

        :param cases: Each instance represents a distinct tag; tag = annotation &#x29FA; category.
                      The frame must include the error matrix frequencies is tp, tn, fp, & fn.
        """

        self.cases = cases

    def precision(self) -> pd.Series:
        """
        Positive Predictive Value

        :return:
        """

        return self.cases.tp.truediv(self.cases.tp + self.cases.fp)

    def sensitivity(self) -> pd.Series:
        """"
        True Positive Rate, Recall, Hit rate

        :return:
        """

        return self.cases.tp.truediv(self.cases.tp + self.cases.fn)

    def specificity(self) -> pd.Series:
        """
        True Negative Rate, Selectivity

        :return:
        """

        return self.cases.tn.truediv(self.cases.tn + self.cases.fp)

    def fscore(self) -> pd.Series:
        """
        F1 Score, F Score, F Measure

        :return:
        """

        ppv = self.precision()
        tpr = self.sensitivity()

        return 2 * (ppv.multiply(tpr)).truediv(ppv + tpr)

    def accuracy(self) -> pd.DataFrame:
        """
        The balanced & standard (imbalanced) accuracies

        :return:
        """

        tpr = self.sensitivity()
        tnr = self.specificity()
        numerator = self.cases[['tp', 'tn']].sum(axis=1)
        denominator = self.cases[['tp', 'fn', 'tn', 'fp']].sum(axis=1)

        return pd.DataFrame(data={'b-accuracy': 0.5*(tpr + tnr),
                                  'i-accuracy': numerator.truediv(denominator)})

    def youden(self) -> pd.Series:
        """
        Youden's J Statistic, Youden's Index

        :return:
        """

        return self.sensitivity() + self.specificity() - 1

    def matthews(self) -> pd.Series:
        """
        Matthews Correlation Coefficient

        :return:
        """

        numerator = self.cases.tp.multiply(self.cases.tn) - self.cases.fp.multiply(self.cases.fn)

        pcp = self.cases.tp + self.cases.fp
        tcp = self.cases.tp + self.cases.fn
        tcn = self.cases.fp + self.cases.tn
        pcn = self.cases.fn + self.cases.tn
        denominator = pcp.multiply(tcp).multiply(tcn).multiply(pcn).pow(0.5)

        return numerator.truediv(denominator)

    def exc(self) -> pd.DataFrame:
        """

        :return:
        """

        data = self.cases
        data.loc[:, 'precision'] = self.precision()
        data.loc[:, 'sensitivity'] = self.sensitivity()
        data.loc[:, 'specificity'] = self.specificity()
        data.loc[:, 'fnr'] = 1 - data['sensitivity']
        data.loc[:, 'fpr'] = 1 - data['specificity']
        data.loc[:, 'f-score'] = self.fscore()
        data.loc[:, 'youden'] = self.youden()
        data.loc[:, 'matthews'] = self.matthews()

        data = pd.concat((data, self.accuracy()), axis=1)
        data.fillna(value=0, inplace=True)

        return data
