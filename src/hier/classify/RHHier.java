package hier.classify;

import java.util.Map;

import data.entry.DProblem;
import data.entry.vectorOperation;

public class RHHier extends WikiSmall{
	
	public void train(DProblem prob, double c){
		this.resetParam(prob);
		double change = Double.POSITIVE_INFINITY;
		while(change>0.001){
			double norm = 0;
			double diff = 0;
			for(Map.Entry<String, Node> nodeEntry : this.nodeMap.entrySet()){
				Node node = nodeEntry.getValue();
				String nodeName = nodeEntry.getKey();
				if(node.childSet.size()>0){
					if(node.parentName!=null){
						Node parentNode = this.nodeMap.get(node.parentName);
						double[] w = vectorOperation.copyOf(parentNode.w);
						for(Node childNode : node.childSet){
							double[] cw = childNode.w;
							for(int j=0;j<w.length;j++){
								w[j] = w[j]+cw[j];
							}
						}
						for(int j=0;j<w.length;j++){
							w[j] = w[j]/(node.childSet.size()+1);
						}
						norm = norm + vectorOperation.dotProd(node.w, node.w);
						double[] sub = vectorOperation.sub(w, node.w);
						diff = diff + vectorOperation.dotProd(sub, sub);
						node.w = w;
					}
				}else{
					double[] w = SolverRank_CS.solve(prob, c, node.alpha, nodeName);
					norm = norm + vectorOperation.dotProd(node.w, node.w);
					double[] sub = vectorOperation.sub(w, node.w);
					diff = diff + vectorOperation.dotProd(sub, sub);
					node.w = w;
					this.labelMap.get(nodeName).w = w;
				}
			}
			change = diff/norm;
//			System.out.println(change);
		}
	}
	
	public void resetParam(DProblem prob){
		for(Map.Entry<String, Node> nodeEntry : this.nodeMap.entrySet()){
			Node node = nodeEntry.getValue();
			node.w = new double[prob.featureNum];
			if(node.childSet.size()==0){
				node.alpha = new double[prob.trainNum];
			}
		}
	}
	
	public void crossValidation(DProblem prob, int k){
		double[] cArray = {0.5,1.0,1.5,2,2.5,3,3.5,4,4.5};
		double maxMicroF1 = 0;
		double optC = 0.5;
		for(double c : cArray){
			double microF1 = 0;
			double macroF1 = 0;
			for(int i=0;i<k;i++){
				prob.merge();
				prob.split(k);
				this.train(prob, c);
				String[][] validO = this.flatPredict(prob.validx);
				microF1 = microF1 + this.calMicroF1(prob.validy, validO);
				macroF1 = macroF1 + this.calMacroF1(prob.validy, validO);
			}
			microF1 = microF1/k;
			macroF1 = macroF1/k;
			if(maxMicroF1<microF1){
				maxMicroF1 = microF1;
				optC = c;
			}
//			System.out.println("c = "+c+" microF1 = " + microF1 +" maroF1 = " + macroF1);
		}
		System.out.println(optC);
		prob.merge();
		this.train(prob, optC);
	}
}
