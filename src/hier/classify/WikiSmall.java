package hier.classify;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Queue;
import java.util.Set;

import data.entry.DProblem;
import data.entry.Feature;
import data.entry.vectorOperation;

public class WikiSmall {
	Map<String, Node> nodeMap;
	Map<String, LabelInfor> labelMap;
	public String rootName = "39348";
	Node[][] levelArray;
	public int levelNum;
	public final int minNum = 1;
	public void loadStruct(String fileName) throws IOException{
		BufferedReader br = new BufferedReader(new FileReader(fileName));
		this.labelMap = new HashMap<String, LabelInfor>();
		this.nodeMap = new HashMap<String, Node>();
		String line = null;
		while((line=br.readLine())!=null){
			String[] items = line.split(" ");
			for(int i=0;i<items.length;i++){
				String nodeName = items[i];
				Node node = this.nodeMap.get(nodeName);
				if(node==null){
					node = new Node(nodeName);
					this.nodeMap.put(nodeName, node);
				}
				if(i>0){
					node.parentName = items[i-1];
					Node parentNode = this.nodeMap.get(items[i-1]);
					parentNode.addChild(node);
				}
				if(i==items.length-1){
					this.labelMap.put(items[i], new LabelInfor(nodeName));
				}
			}
		}
		System.out.println("nodeNum = " + this.nodeMap.size() +" labelNum = " + this.labelMap.size());
		Set<Node> nodeSet = new HashSet<Node>();
		for(Map.Entry<String, Node> nodeEntry : this.nodeMap.entrySet()){
			Node node = nodeEntry.getValue();
			if(node.parentName==null){
				nodeSet.add(node);
			}
		}
		if(nodeSet.size()>1){
			Node rootNode = new Node(this.rootName);
			for(Node node : nodeSet){
				node.parentName = this.rootName;
				rootNode.addChild(node);
			}
			this.nodeMap.put(this.rootName, rootNode);
		}else{
			this.rootName = nodeSet.iterator().next().nodeName;
		}
		br.close();
	}
	
	public void nodeDepth(){
		Queue<Node> nodeQueue = new LinkedList<Node>();
		nodeQueue.add(this.nodeMap.get(this.rootName));
		while(!nodeQueue.isEmpty()){
			Node node = nodeQueue.poll();
			for(Node childNode : node.childSet){
				childNode.depth = node.depth+1;
				nodeQueue.offer(childNode);
			}
		}
	}
	
	public void conLevelArray(){
		List<Node[]> levelList = new ArrayList<>();
		List<Node> nodeList = new ArrayList<Node>();
		Node root = this.nodeMap.get(this.rootName);
		nodeList.addAll(root.childSet);
		while(!nodeList.isEmpty()){
			List<Node> nNodeList = new ArrayList<Node>();
			Node[] nodeArray = new Node[nodeList.size()];
			for(int i=0;i<nodeList.size();i++){
				Node node = nodeList.get(i);
				nodeArray[i] = nodeList.get(i);
				for(Node childNode : node.childSet){
					nNodeList.add(childNode);
				}
			}
			levelList.add(nodeArray);
			nodeList = nNodeList;
		}
		this.levelNum = levelList.size();
		this.levelArray = new Node[levelList.size()][];
		for(int i=0;i<levelList.size();i++){
			this.levelArray[i] = levelList.get(i);
		}
	}
	
	public void assiginWeight(DProblem prob){
		for(String[] yi : prob.y){
			for(Map.Entry<String, Node> nodeEntry : this.nodeMap.entrySet()){
				Node node = nodeEntry.getValue();
				String nodeName = nodeEntry.getKey();
				if(this.isRelate(yi, nodeName)){
					node.insNum++;
				}
			}
		}
		for(Map.Entry<String, Node> nodeEntry : this.nodeMap.entrySet()){
			Node node = nodeEntry.getValue();
//			System.out.println(node.nodeName +" insNum " + node.insNum);
			for(Node childNode : node.childSet){
				double weight = childNode.insNum/(node.insNum-childNode.insNum);
				childNode.weight = weight;
			}
		}
	}
	
	public void shrinkTrain(DProblem prob, double c1, double c2){
		DProblem nodePob = prob;
		String[][] o = null;
		for(int i=0;i<this.levelNum;i++){
			long start = System.currentTimeMillis();
			Node[] nodeArray = this.levelArray[i];
			for(Node node : nodeArray){
				nodePob = this.conNodeProb(prob, o, node.parentName, node.nodeName);
				node.bw = SolverRank_CS.biasedSolve(nodePob, c1, node.nodeName);
				node.w = SolverRank_CS.solve(nodePob, c2, node.nodeName);
			}
			o = this.calLevelCad(o, nodeArray, prob.x);
			long time = System.currentTimeMillis()-start;
//			System.out.println("level = " + i + " " + time);
		}
	}
	
	public void topDownTrain(DProblem prob, double c){
		Queue<Node> nodeQueue = new LinkedList<Node>();
		nodeQueue.offer(this.nodeMap.get(this.rootName));
		while(!nodeQueue.isEmpty()){
			Node node = nodeQueue.poll();
			DProblem nodeProb = this.conTPDNodeProb(prob, node.nodeName);
			for(Node childNode : node.childSet){
				childNode.w = SolverRank_CS.solve(nodeProb, c, childNode.nodeName);
				nodeQueue.offer(childNode);
			}
		}
	}
	public void flatTrain(DProblem prob, double c){
		for(Map.Entry<String, LabelInfor> labelEntry : this.labelMap.entrySet()){
			LabelInfor labelInfor = labelEntry.getValue();
			String labelName = labelEntry.getKey();
			labelInfor.w = SolverRank_CS.solve(prob, c, labelName);
		}
	}
	public void treeLossTrain(DProblem prob, double c){
		this.assiginWeight(prob);
		for(Map.Entry<String, LabelInfor> labelEntry : this.labelMap.entrySet()){
			LabelInfor labelInfor = labelEntry.getValue();
			String labelName = labelEntry.getKey();
			double[] delta = this.calDelta(prob.y, labelName);
			labelInfor.w = SolverRank_CS.solve(prob, c, labelName, delta);
		}
	}
	
	
	public void crossValidation(DProblem prob, int k){
		double[] cArray = {0.5,1.0,1.5,2,2.5,3,3.5,4,4.5};
		double maxMicroF1 = 0;
		double optC = 0.5;
		for(double c : cArray){
			double averageMicroF1 = 0;
			double averageMacroF1 = 0;
			for(int i=0;i<k;i++){
				prob.split(k);
				this.shrinkTrain(prob, c, c);
				String[][] valido = this.calPredict(prob.validx);
				averageMacroF1 = averageMacroF1 + this.calMacroF1(prob.validy, valido);
				averageMicroF1 = averageMicroF1 + this.calMicroF1(prob.validy, valido);
				prob.merge();
			}
			averageMacroF1 = averageMacroF1/k;
			averageMicroF1 = averageMicroF1/k;
			if(averageMicroF1>maxMicroF1){
				maxMicroF1 = averageMicroF1;
				optC = c;
			}
//			System.out.println("c = " + c +" microF1 = " + averageMicroF1 +" macroF1 = " + averageMacroF1);
		}
		prob.merge();
		System.out.println("optc = " + optC);
		this.shrinkTrain(prob, optC, optC);
	}
	public void TPDcrossValidation(DProblem prob, int k){
		double[] cArray = {0.5,1.0,1.5,2,2.5,3,3.5,4,4.5};
		double maxMicroF1 = 0;
		double optC = 0.5;
		prob.merge();
		for(double c : cArray){
			double averageMicroF1 = 0;
			double averageMacroF1 = 0;
			for(int i=0;i<k;i++){
				prob.split(k);
				this.topDownTrain(prob, c);
				String[][] valido = this.topDownPredict(prob.validx);
				averageMacroF1 = averageMacroF1 + this.calMacroF1(prob.validy, valido);
				averageMicroF1 = averageMicroF1 + this.calMicroF1(prob.validy, valido);
				prob.merge();
			}
			averageMacroF1 = averageMacroF1/k;
			averageMicroF1 = averageMicroF1/k;
			if(averageMicroF1>maxMicroF1){
				maxMicroF1 = averageMicroF1;
				optC = c;
			}
//			System.out.println("c = " + c +" microF1 = " + averageMicroF1 +" macroF1 = " + averageMacroF1);
		}
		System.out.println("optc = " + optC);
		this.topDownTrain(prob, optC);
	}
	
	public void FLTcrossValidation(DProblem prob, int k){
		double[] cArray = {0.5,1.0,1.5,2,2.5,3,3.5,4,4.5};
		double maxMicroF1 = 0;
		double optC = 0.5;
		for(double c : cArray){
			double averageMicroF1 = 0;
			double averageMacroF1 = 0;
			for(int i=0;i<k;i++){
				prob.merge();
				prob.split(k);
				this.flatTrain(prob, c);
				String[][] valido = this.flatPredict(prob.validx);
				averageMacroF1 = averageMacroF1 + this.calMacroF1(prob.validy, valido);
				averageMicroF1 = averageMicroF1 + this.calMicroF1(prob.validy, valido);
			}
			averageMacroF1 = averageMacroF1/k;
			averageMicroF1 = averageMicroF1/k;
			if(averageMicroF1>maxMicroF1){
				maxMicroF1 = averageMicroF1;
				optC = c;
			}
//			System.out.println("c = " + c +" microF1 = " + averageMicroF1 +" macroF1 = " + averageMacroF1);
		}
		prob.merge(); 
		System.out.println("optc = " + optC);
		this.flatTrain(prob, optC);
	}
	public void TLcrossValidation(DProblem prob, int k){
		double[] cArray = {0.5,1.0,1.5,2,2.5,3,3.5,4,4.5};
		double maxMicroF1 = 0;
		double optC = 0.5;
		for(double c : cArray){
			double averageMicroF1 = 0;
			double averageMacroF1 = 0;
			for(int i=0;i<k;i++){
				prob.merge();
				prob.split(k);
				this.treeLossTrain(prob, c);
				String[][] valido = this.flatPredict(prob.validx);
				averageMacroF1 = averageMacroF1 + this.calMacroF1(prob.validy, valido);
				averageMicroF1 = averageMicroF1 + this.calMicroF1(prob.validy, valido);
			}
			averageMacroF1 = averageMacroF1/k;
			averageMicroF1 = averageMicroF1/k;
			if(averageMicroF1>maxMicroF1){
				maxMicroF1 = averageMicroF1;
				optC = c;
			}
//			System.out.println("c = " + c +" microF1 = " + averageMicroF1 +" macroF1 = " + averageMacroF1);
		}
		prob.merge(); 
		System.out.println("optc = " + optC);
		this.treeLossTrain(prob, optC);
	}
	
	public String[][] flatPredict(Feature[][] x){
		int sampleNum = x.length;
		String[][] o = new String[sampleNum][];
		for(int i=0;i<sampleNum;i++){
			List<String> list = new ArrayList<String>();
			double maxValue = Double.NEGATIVE_INFINITY;
			String maxLabel = null;
			for(Map.Entry<String, LabelInfor> labelEntry : this.labelMap.entrySet()){
				LabelInfor labelInfor = labelEntry.getValue();
				String labelName = labelEntry.getKey();
				double oil = vectorOperation.dotProd(labelInfor.w, x[i]);
				if(oil>0){
					list.add(labelName);
				}
				if(oil>maxValue){
					maxLabel = labelName;
					maxValue = oil;
				}
			}
//			o[i] = new String[list.size()];
//			for(int l=0;l<list.size();l++){
//				o[i][l] = list.get(l);
//			}
			if(list.size()>0){
				o[i] = new String[list.size()];
				for(int l=0;l<list.size();l++){
					o[i][l] = list.get(l);
				}
			}else{
				o[i] = new String[1];
				o[i][0] = maxLabel;
			}
		}
		return o;
	}
	
	
	private DProblem conTPDNodeProb(DProblem prob, String parenName) {
		DProblem nodeProb = new DProblem();
		List<Feature[]> sampleList = new ArrayList<Feature[]>();
		List<String[]> labelList = new ArrayList<String[]>();
		Node parentNode = this.nodeMap.get(parenName);
		for(int i=0;i<prob.trainNum;i++){
			if(this.isRelate(prob.y[i], parenName)){
				List<String> list = new ArrayList<String>();
				for(Node node : parentNode.childSet){
					if(this.isRelate(prob.y[i], node.nodeName)){
						list.add(node.nodeName);
					}
				}
				String[] yi = new String[list.size()];
				for(int l=0;l<list.size();l++){
					yi[l] = list.get(l);
				}
				sampleList.add(prob.x[i]);
				labelList.add(yi);
			}
		}
		Feature[][] x = new Feature[sampleList.size()][];
		String[][] y = new String[labelList.size()][];
		for(int i=0;i<sampleList.size();i++){
			x[i] = sampleList.get(i);
			y[i] = labelList.get(i);
		}
		nodeProb.bias = prob.bias;
		nodeProb.featureNum = prob.featureNum;
		nodeProb.trainNum = x.length;
		nodeProb.x = x;
		nodeProb.y = y;
		nodeProb.sampleList = sampleList;
		nodeProb.labelList = labelList;
		return nodeProb;
	}
	
	public String[][] topDownPredict(Feature[][] x){
		int sampleNum = x.length;
		String[][] o = new String[sampleNum][];
		Node root = this.nodeMap.get(this.rootName);
		for(int i=0;i<sampleNum;i++){
			int minLabel = this.minNum;
			Queue<Node> nodeQueue = new LinkedList<Node>();
			List<String> list = new ArrayList<String>();
			nodeQueue.add(root);
			while(!nodeQueue.isEmpty()){
				Node node = nodeQueue.poll();
				Node maxNode = null;
				double maxValue = Double.NEGATIVE_INFINITY;
				for(Node childNode : node.childSet){
					double oil = vectorOperation.dotProd(childNode.w, x[i]);
					if(oil>0){
						if(childNode.childSet.size()==0){
							list.add(childNode.nodeName);
							minLabel = 0;
						}else{
							nodeQueue.add(childNode);
						}
					}
					if(oil>maxValue){
						maxValue = oil;
						maxNode = childNode;
					}
				}
				if(minLabel>0 && nodeQueue.isEmpty()){
					if(maxNode.childSet.size()==0){
						list.add(maxNode.nodeName);
						minLabel = 0;
					}else{
						nodeQueue.offer(maxNode);
					}
				}
			}
			o[i] = new String[list.size()];
			for(int l=0;l<list.size();l++){
				o[i][l] = list.get(l);
			}
		}
		return o;
	}

	public String[][] calLevelCad(String[][] preo, Node[] nodeArray, Feature[][] x){
		int sampleNum = x.length;
		String[][] o = new String[sampleNum][];
		for(int i=0;i<sampleNum;i++){
			List<String> labelList = new ArrayList<String>();
			for(Node node : nodeArray){
				if(preo ==null || this.contains(preo[i], node.parentName)){
					double oil = vectorOperation.dotProd(node.bw, x[i]);
					if(oil>-1){
						labelList.add(node.nodeName);
					}
				}
				o[i] = new String[labelList.size()];
				for(int l=0;l<labelList.size();l++){
					o[i][l] = labelList.get(l);
				}
			}
		}
		return o;
	}
	
	
	public String[][] calPredict(Feature[][] x){
		ScoreComparator c = new ScoreComparator();
		int sampleNum = x.length;
		String[][] o = new String[sampleNum][];
		for(int i=0;i<sampleNum;i++){
			List<String> labelList = new ArrayList<String>();
			Set<String> nodeSet = new HashSet<String>();
			nodeSet.add(this.rootName);
			int minLabel = this.minNum;;
			for(int level = 0;level<this.levelNum;level++){
				Map<String, Double> scoreMap = new HashMap<String, Double>();
				Map<String, Double> outMap = new HashMap<String, Double>();
				int cadNum = 0;
				int outNum = 0;
				for(Node node : this.levelArray[level]){
					if(nodeSet.contains(node.parentName)){
						double oil = vectorOperation.dotProd(node.w, x[i]);
						double boil = vectorOperation.dotProd(node.bw, x[i]);
						if(oil>0){
							outNum++;
						}
						if(boil>-1){
							cadNum++;
						}
						outMap.put(node.nodeName, oil);
						scoreMap.put(node.nodeName, boil);
					}
				}
				List<Map.Entry<String, Double>> scoreList = new ArrayList<Map.Entry<String, Double>>(scoreMap.entrySet());
				List<Map.Entry<String, Double>> outList = new ArrayList<Map.Entry<String, Double>>(outMap.entrySet());
				Collections.sort(scoreList, c);
				Collections.sort(outList, c);
				if(labelList.size()>0){
					minLabel = 0;
				}else{
					minLabel = this.minNum;
				}
				int len = Math.max(minLabel, outNum);
//				int len = outNum;
				len = Math.min(len, outList.size());
				for(int l=0;l<len;l++){
					String nodeName = scoreList.get(l).getKey();
					Node node = this.nodeMap.get(nodeName);
					if(node.childSet.size()==0){
						labelList.add(nodeName);
						minLabel = 0;
					}
				}
//				len = Math.max(minLabel, cadNum);
				len = cadNum;
				len = Math.min(len, scoreList.size());
				for(int l=0;l<len;l++){
					String nodeName = scoreList.get(l).getKey();
					nodeSet.add(nodeName);
				}
				
			}
			o[i] = new String[labelList.size()];
			for(int l=0;l<labelList.size();l++){
				o[i][l] = labelList.get(l);
			}
		}
		return o;
	}
	public String[][] SLPredict(Feature[][] x){
		int sampleNum = x.length;
		ScoreComparator c = new ScoreComparator();
		String[][] o = new String[sampleNum][];
		for(int i=0;i<sampleNum;i++){
			Set<String> nodeSet = new HashSet<String>();
			nodeSet.add(this.rootName);
			for(int level = 0;level<this.levelNum;level++){
				Map<String, Double> scoreMap = new HashMap<String, Double>();
				int cadNum = 0;
				double maxValue = Double.NEGATIVE_INFINITY;
				String maxLable = null;
				for(Node node : this.levelArray[level]){
					if(nodeSet.contains(node.parentName)){
						double oil = vectorOperation.dotProd(node.w, x[i]);
						double boil = vectorOperation.dotProd(node.bw, x[i]);
						if(boil>0){
							cadNum++;
						}
						scoreMap.put(node.nodeName, boil);
						if(oil>maxValue){
							maxValue = oil;
							if(node.childSet.size()==0){
								maxLable = node.nodeName;
							}else{
								maxLable=null;
							}
						}
					}
				}
				if(maxLable!=null){
 					o[i] = new String[1];
					o[i][0] = maxLable;
					break;
				}else{
					nodeSet = new HashSet<String>();
					List<Map.Entry<String, Double>> scoreList = new ArrayList<Map.Entry<String, Double>>(scoreMap.entrySet());
					Collections.sort(scoreList, c);
					int len = Math.max(this.minNum, cadNum);
					for(int l=0;l<len;l++){
						nodeSet.add(scoreList.get(l).getKey());
					}
				}
			}
		}
		return o;
	}
	
	public String[][] flatSLPredict(Feature[][] x){
		int sampleNum = x.length;
		String[][] o = new String[sampleNum][];
		for(int i=0;i<sampleNum;i++){
			String maxLabel = null;
			double maxValue = Double.NEGATIVE_INFINITY;
			for(Map.Entry<String, LabelInfor> labelEntry : this.labelMap.entrySet()){
				String labelName = labelEntry.getKey();
				LabelInfor labelInfor = labelEntry.getValue();
				double oil = vectorOperation.dotProd(labelInfor.w, x[i]);
				if(oil>maxValue){
					maxLabel = labelName;
					maxValue = oil;
				}
			}
			o[i] = new String[1];
			o[i][0] = maxLabel;
		}
		return o;
	}
	
	private DProblem conNodeProb(DProblem prob, String[][] o, String parentName, String nodeName) {
		DProblem nodeProb = new DProblem();
		List<Feature[]> sampleList = new ArrayList<Feature[]>();
		List<String[]> labelList = new ArrayList<String[]>();
		for(int i=0;i<prob.trainNum;i++){
			if(o==null || this.contains(o[i], parentName)){
				String[] yi = new String[1];
				if(this.isRelate(prob.y[i], nodeName)){
					yi[0] = nodeName;
				}else{
					yi[0] = "null";
				}
				sampleList.add(prob.x[i]);
				labelList.add(yi);
			}
		}
		Feature[][] x = new Feature[sampleList.size()][];
		String[][] y = new String[sampleList.size()][];
		for(int i=0;i<x.length;i++){
			x[i] = sampleList.get(i);
			y[i] = labelList.get(i);
		}
		nodeProb.featureNum = prob.featureNum;
		nodeProb.trainNum = sampleList.size();
		nodeProb.x = x;
		nodeProb.y = y;
		return nodeProb;
	}
	public boolean contains(String[] yi, String labelName){
		if(labelName.equals(this.rootName)){
			return true;
		}
		for(String yil : yi){
			if(labelName.equals(yil)){
				return true;
			}
		}
		return false;
	}
	public boolean isRelate(String[] yi, String nodeName){
		for(String yil : yi){
			String pathName = yil;
			while(pathName!=null){
				if(pathName.equals(nodeName)){
					return true;
				}
//				System.out.println(pathName);
				pathName = this.nodeMap.get(pathName).parentName;
			}
		}
		return false;
	}
	
	public double[] calDelta(String[][] y, String labelName){
		double deltaMax = 0.5;
		String pathName = labelName;
		while(!rootName.equals(pathName)){
			deltaMax++;
			pathName = this.nodeMap.get(pathName).parentName;
		}
		double[] delta = new double[y.length];
		for(int i=0;i<y.length;i++){
			if(this.contains(y[i], labelName)){
				delta[i] = deltaMax;
			}else{
				pathName = labelName;
				while(!this.isRelate(y[i], pathName)){
					Node node = this.nodeMap.get(pathName);
					delta[i] = delta[i] + node.weight;
					pathName = node.parentName;
				}
				delta[i] = delta[i] + 0.5;
			}
		}
		return delta;
	}
	
	
	public double calMicroF1(String[][] y, String[][] o){
		double TP = 0;
		double FP = 0;
		double FN = 0;
		double microF1 = 0;
		int sampelNum = y.length;
		for(Map.Entry<String, LabelInfor> labelEntry : this.labelMap.entrySet()){
			String labelName = labelEntry.getKey();
			for(int i=0;i<sampelNum;i++){
				if(this.contains(y[i], labelName) && this.contains(o[i], labelName)){
					TP++;
				}
				if(!this.contains(y[i], labelName) && this.contains(o[i], labelName)){
					FP++;
				}
				if(this.contains(y[i], labelName) && !this.contains(o[i], labelName)){
					FN++;
				}
			}
		}
		if(TP>0){
			double P = TP/(TP+FP);
			double R = TP/(TP+FN);
			microF1 = 2*P*R/(P+R);
		}
		return microF1;
	}
	
	public double calMacroF1(String[][] y, String[][] o){
		double macroF1 = 0;
		int sampelNum = y.length;
		for(Map.Entry<String, LabelInfor> labelEntry : this.labelMap.entrySet()){
			String labelName = labelEntry.getKey();
			double TPt = 0;
			double FPt = 0;
			double FNt = 0;
			for(int i=0;i<sampelNum;i++){
				if(this.contains(y[i], labelName) && this.contains(o[i], labelName)){
					TPt++;
				}
				if(!this.contains(y[i], labelName) && this.contains(o[i], labelName)){
					FPt++;
				}
				if(this.contains(y[i], labelName) && !this.contains(o[i], labelName)){
					FNt++;
				}
			}
			if (TPt>0) {
				double Pt = TPt / (TPt + FPt);
				double Rt = TPt / (TPt + FNt);
				double f1 = 2 * Pt * Rt / (Pt + Rt);
				macroF1 =macroF1 + f1;
			}
		}
		return macroF1/this.labelMap.size();
	}
}
