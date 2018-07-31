package data.entry;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.Random;

public class DProblem {
	//feature vectors for train and validation data set
		public Feature[][] x;
		public Feature[][] validx;
		
		//label vector for train and validation data set
		public String[][] y;
		public String[][] validy;
		
		public int trainNum;
		public int labelNum;
		public int featureNum;
		
		public double bias;
		
		public List<Feature[]> sampleList;
		public List<String[]> labelList;
		
		public double[] df;
		
		public void loadData(String fileName, boolean isTrain) throws IOException{
			BufferedReader br = new BufferedReader(new FileReader(fileName));
			String line = null;
			this.sampleList = new ArrayList<Feature[]>();
			this.labelList = new ArrayList<String[]>();
			int maxIndex = 0;
			int count = 0;
			while((line=br.readLine())!=null){
				count++;
				if(!"".equals(line)){
					String[] items = line.split(" ");
					String[] labelArray = items[0].split(",");
					this.labelList.add(labelArray);
					int len = items.length-1;
					if(this.bias>0){
						len = len + 1;
					}
					Feature[] xi = new Feature[len];
					for(int i=1;i<items.length;i++){
						String[] kv = items[i].split(":");
						int key = Integer.parseInt(kv[0]);
						double value = Double.parseDouble(kv[1]);
						xi[i-1] = new FeatureNode(key, value);
						maxIndex = Math.max(maxIndex, key);
					}
					this.sampleList.add(xi);
				}
			}
			br.close();
			this.trainNum = this.sampleList.size();
			if (isTrain) {
				this.featureNum = maxIndex + 1;
				if (bias > 0) {
					this.featureNum = this.featureNum + 1;
				} 
			}
			this.x = new Feature[this.trainNum][];
			this.y = new String[this.trainNum][];
			Feature biasFeature = null;
			if(this.bias>0){
				biasFeature = new FeatureNode(this.featureNum-1, this.bias);
			}
			Comparator<Feature> c = new IndexComparator();
			for(int i=0;i<this.sampleList.size();i++){
				this.x[i] = this.sampleList.get(i);
				if(this.bias>0){
					this.x[i][this.x[i].length-1] = biasFeature;
				}
				Arrays.sort(this.x[i], c);
				this.y[i] = this.labelList.get(i);
			}
		}
		
		public void loadData(String[] fileNames, boolean isTrain) throws IOException{
			String line = null;
			this.sampleList = new ArrayList<Feature[]>();
			this.labelList = new ArrayList<String[]>();
			int maxIndex = 0;
			int count = 0;
			for (String fileName : fileNames) {
				BufferedReader br = new BufferedReader(new FileReader(fileName));
				while ((line = br.readLine()) != null) {
					if (!"".equals(line)) {
						count++;
						String[] items = line.split(" ");
						String[] labelArray = items[0].split(",");
						this.labelList.add(labelArray);
						int len = items.length - 1;
						if (this.bias > 0) {
							len = len + 1;
						}
						Feature[] xi = new Feature[len];
						for (int i = 1; i < items.length; i++) {
							String[] kv = items[i].split(":");
							int key = Integer.parseInt(kv[0]);
							double value = Double.parseDouble(kv[1]);
							xi[i - 1] = new FeatureNode(key, value);
							maxIndex = Math.max(maxIndex, key);
						}
						this.sampleList.add(xi);
					}
				}
				br.close();
			}
			System.out.println(maxIndex);
			this.trainNum = this.sampleList.size();
			if (isTrain) {
				this.featureNum = maxIndex + 1;
				if (bias > 0) {
					this.featureNum = this.featureNum + 1;
				} 
			}
			this.x = new Feature[this.trainNum][];
			this.y = new String[this.trainNum][];
			Feature biasFeature = null;
			if(this.bias>0){
				biasFeature = new FeatureNode(this.featureNum-1, this.bias);
			}
			Comparator<Feature> c = new IndexComparator();
			for(int i=0;i<this.sampleList.size();i++){
				this.x[i] = this.sampleList.get(i);
				if(this.bias>0){
					this.x[i][this.x[i].length-1] = biasFeature;
				}
				Arrays.sort(this.x[i], c);
				this.y[i] = this.labelList.get(i);
			}
		}
		
		
		
		
		public double[] callabelVec(String labelName){
			double[] yl = new double[this.trainNum];
			for(int i=0;i<this.trainNum;i++){
				double value = -1;
				for(String label : this.y[i]){
					if(label.equals(labelName)){
						value = 1;
						break;
					}
				}
				yl[i] = value;
			}
			return yl;
		}
		
		public double[] caltopDownLableVec(String labelName, String parentName, double[] w){
			double[] yl = new double[this.trainNum];
			for(int i=0;i<this.trainNum;i++){
				double value = 0;
				for(String label : this.y[i]){
					if(label.equals(parentName)){
						value = -1;
						break;
					}
				}
				double dotProd = vectorOperation.dotProd(w, this.x[i]);
				if(dotProd>0){
					value = -1;
				}
				for(String label : this.y[i]){
					if(label.equals(labelName)){
						value = 1;
						break;
					}
				}
				yl[i] = value;
			}
			return yl;
		}
		
		public void split(int k){
			int validSzie = this.sampleList.size()/k;
			if(validSzie==0){
				validSzie = 1;
			}
			Random rand = new Random();
			this.validx = new Feature[validSzie][];
			this.validy = new String[validSzie][];
			for(int i=0;i<validSzie;i++){
				int index = rand.nextInt(sampleList.size());
				this.validx[i] = this.sampleList.remove(index);
				this.validy[i] = this.labelList.remove(index);
			}
			this.trainNum = this.sampleList.size();
			this.x = new Feature[this.trainNum][];
			this.y = new String[this.trainNum][];
			for(int i=0;i<this.sampleList.size();i++){
				this.x[i] = this.sampleList.get(i);
				this.y[i] = this.labelList.get(i);
			}
		}
		
		public void merge(){
			this.sampleList = new ArrayList<Feature[]>();
			this.labelList = new ArrayList<String[]>();
			for(int i=0;i<this.x.length;i++){
				this.sampleList.add(this.x[i]);
				this.labelList.add(this.y[i]);
			}
			if(validx!=null){
				for(int i=0;i<this.validx.length;i++){
					this.sampleList.add(this.validx[i]);
					this.labelList.add(this.validy[i]);
				}
			}
			this.trainNum = this.sampleList.size();
			this.x = new Feature[this.trainNum][];
			this.y = new String[this.trainNum][];
			for(int i=0;i<this.trainNum;i++){
				this.x[i] = this.sampleList.get(i);
				this.y[i] = this.labelList.get(i);
			}
		}
		
		public void normalize(){
			this.df = new double[this.featureNum];
			for(Feature[] xi : this.x){
				for(Feature f : xi){
					int index = f.getIndex();
					this.df[index] += 1;
				}
			}
			for(Feature[] xi : this.x){
				this.normalize(xi);
			}
		}
		
		public void normalize(Feature[][] data){
			for(Feature[] xi : data){
				this.normalize(xi);
			}
		}
		
		public void normalize(Feature[] xi){
			double norm = 0;
			for(Feature f : xi){
				int index = f.getIndex();
				double value = f.getValue();
//				if(value==0 || this.df[index]==0){
//					System.out.println("index = " + index +" value = " + value);
//				}
				if(value!=0){
					value = (1+Math.log(value))*Math.log(this.trainNum/this.df[index]);
				}
				if(this.bias<=0 || index!=this.featureNum-1){
					f.setValue(value);
					norm += value*value;
				}
			}
			norm = Math.sqrt(norm);
			for(Feature f : xi){
				double value = f.getValue()/norm;
				if(this.bias<=0 || f.getIndex()!=this.featureNum-1){
					f.setValue(value);
				}
			}
		}
		
		public void outputDta(String fileName) throws Exception{
			BufferedWriter bw = new BufferedWriter(new FileWriter(fileName));
			for(int i=0;i<this.trainNum;i++){
				for(int l=0;l<this.y[i].length;l++){
					bw.write(this.y[i][l]);
					if(l!=(this.y[i].length-1)){
						bw.write(",");
					}
				}
				for(Feature f: this.x[i]){
					int index = f.getIndex();
					double value = f.getValue();
					if(index>=0 && index<this.featureNum){
						bw.write(" "+index +":"+value);
					}
				}
				bw.newLine();
			}
			bw.close();
		}
		
		public double calAverageLables(){
			double totalLabelNum = 0;
			for(int i=0;i<this.y.length;i++){
				totalLabelNum += this.y[i].length;
			}
			return totalLabelNum/this.trainNum;
		}
}
