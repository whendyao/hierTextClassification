package hier.classify.test;

import java.io.IOException;

import org.junit.Test;

import data.entry.DProblem;
import hier.classify.RHHier;



public class RHSVMTest {
	@Test
	public void fixparamTest() throws IOException{
		RHHier hier = new RHHier();
		hier.loadStruct("E:\\�ı��������Ͽ�\\�½��ļ���\\data\\hier.txt");
		hier.rootName = "null";
		DProblem prob = new DProblem();
		prob.bias = 1;
		prob.loadData("E:\\�ı��������Ͽ�\\�½��ļ���\\data\\train3.svm", true);
		
		hier.train(prob, 1);
		DProblem testProb = new DProblem();
		testProb.bias = prob.bias;
		testProb.featureNum = prob.featureNum;
		testProb.loadData("E:\\�ı��������Ͽ�\\�½��ļ���\\data\\test3.svm", false);
		String[][] o = hier.flatPredict(testProb.x);
		double microF1 = hier.calMicroF1(testProb.y, o);
		double macroF1 = hier.calMacroF1(testProb.y, o);
		System.out.println("microF1 = " + microF1 +" macroF1 = " + macroF1);
	}
	
	@Test
	public void crossvalidation() throws Exception{
		RHHier hier = new RHHier();
		hier.loadStruct("E:\\�ı��������Ͽ�\\�½��ļ���\\data\\hier.txt");
		hier.rootName = "null";
		for(int i=1;i<6;i++){
			DProblem prob = new DProblem();
			prob.bias = 1;
			prob.loadData("E:\\�ı��������Ͽ�\\�½��ļ���\\data\\train"+i+".svm", true);
			prob.outputDta("E:\\�ı��������Ͽ�\\DMOZMultilabel\\train_ltc.svm");
//			hier.levelNum = 2;
			
			DProblem testProb = new DProblem();
			testProb.bias = prob.bias;
			testProb.featureNum = prob.featureNum;
			testProb.loadData("E:\\�ı��������Ͽ�\\�½��ļ���\\data\\test"+i+".svm", true);
			hier.crossValidation(prob, 3);
			String[][] o = hier.flatPredict(testProb.x);
			double microF1 = hier.calMicroF1(testProb.y, o);
			double macroF1 = hier.calMacroF1(testProb.y, o);
			System.out.println("microF1 = " + microF1 +" macroF1 = " + macroF1);
		}
	}
}
