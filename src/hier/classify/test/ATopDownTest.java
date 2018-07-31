package hier.classify.test;

import org.junit.Test;

import hier.classify.WikiSmall;
import lib.linear.DProblem;

public class ATopDownTest {
	@Test
	public void fixParamTest() throws Exception{
		WikiSmall hier = new WikiSmall();
		hier.loadStruct("E:\\�ı��������Ͽ�\\�½��ļ���\\data\\hier.txt");
		hier.conLevelArray();
		DProblem prob = new DProblem();
		prob.bias = 1;
		prob.loadData("E:\\�ı��������Ͽ�\\�½��ļ���\\data\\train3.svm", true);
		
		hier.shrinkTrain(prob, 1.5,1.5);
		
		DProblem testProb = new DProblem();
		testProb.bias = prob.bias;
		testProb.featureNum = prob.featureNum;
		testProb.loadData("E:\\�ı��������Ͽ�\\�½��ļ���\\data\\test3.svm", false);
		
		String[][] testo = hier.calPredict(testProb.x);
		double microF1 = hier.calMicroF1(testProb.y, testo);
		double macroF1 = hier.calMacroF1(testProb.y, testo);
		System.out.println("microF1 = " + microF1 + " macroF1 = " + macroF1);
	}
	@Test
	public void crossValidation() throws Exception{
		WikiSmall hier = new WikiSmall();
		hier.loadStruct("E:\\�ı��������Ͽ�\\�½��ļ���\\data\\hier.txt");
		hier.conLevelArray();
		for(int i=1;i<6;i++){
			DProblem prob = new DProblem();
			prob.bias = 1;
			prob.loadData("E:\\�ı��������Ͽ�\\�½��ļ���\\data\\train"+i+".svm", true);
			
			hier.crossValidation(prob, 3);
			
			DProblem testProb = new DProblem();
			testProb.bias = prob.bias;
			testProb.featureNum = prob.featureNum;
			testProb.loadData("E:\\�ı��������Ͽ�\\�½��ļ���\\data\\test"+i+".svm", false);
			
			String[][] testo = hier.calPredict(testProb.x);
			double microF1 = hier.calMicroF1(testProb.y, testo);
			double macroF1 = hier.calMacroF1(testProb.y, testo);
			System.out.println("microF1 = " + microF1 + " macroF1 = " + macroF1);
		}
	}
}
