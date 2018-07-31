package data.entry;

import java.util.ArrayList;
import java.util.List;

public class vectorOperation {
	public static double dotProd(double[] x1, Feature[] x2){
		double dotProd = 0;
		for(Feature feature : x2){
			int index = feature.getIndex();
			double value = feature.getValue();
			dotProd = dotProd + value*x1[index];
		}
		return dotProd;
	}
	
	public static double dis(double[] x1, double[] x2){
		double result = 0;
		for(int i=0;i<x1.length;i++){
			result = result + (x1[i]-x2[i])*(x1[i]-x2[i]);
		}
		return Math.sqrt(result);
	}
	
	public static double dotProd(Feature[] x1, Feature[] x2){
		double dotProd = 0;
		int i=0;
		int j=0;
		while(i<x1.length && j<x2.length){
			int x1Index = x1[i].getIndex();
			int x2Index = x2[j].getIndex();
			if(x1Index==x2Index){
				dotProd = dotProd + x1[i].getValue()*x2[j].getValue();
				i++;
				j++;
			}else if (x1Index>x2Index){
				j++;
			}else{
				i++;
			}
		}
		return dotProd;
	}
	
	public static double dotProd(double[] x1, double[] x2){
		double dotProd = 0;
		for(int i=0;i<x1.length;i++){
			dotProd = dotProd + x1[i]*x2[i];
		}
		return dotProd;
	}
	
	public static double[] copyOf(double[] originVector){
		double[] copy = new double[originVector.length];
		System.arraycopy(originVector, 0, copy, 0, originVector.length);
		return copy;
	}
	
	public static int[] copyOf(int[] originVector){
		int[] copy = new int[originVector.length];
		System.arraycopy(originVector, 0, copy, 0, originVector.length);
		return copy;
	}
	
	public static double[] sum(Feature[] x1, double[] x2){
		for(Feature feature : x1){
			int index = feature.getIndex();
			double value = feature.getValue();
			x2[index] = x2[index] + value; 
		}
		return x2;
	}
	
	public static double[] sub(Feature[] x1, double[] x2){
		for(Feature feature : x1){
			int index = feature.getIndex();
			double value = feature.getValue();
			x2[index] = x2[index] - value; 
		}
		return x2;
	}
	
	public static Feature[] sum(Feature[] x1, Feature[] x2){
		List<Feature> featureList = new ArrayList<Feature>();
		int x1i = 0;
		int x2i = 0;
		while(x1i<x1.length && x2i<x2.length){
			Feature x1Feature = x1[x1i];
			Feature x2Feature = x2[x2i];
			int x1Index = x1Feature.getIndex();
			int x2Index = x2Feature.getIndex();
			double x1Value = x1Feature.getValue();
			double x2Value = x2Feature.getValue();
			if(x1Index>x2Index){
				Feature feature = new FeatureNode(x2Index, x2Value);
				featureList.add(feature);
				x2i++;
			}
			if(x1Index<x2Index){
				Feature feature = new FeatureNode(x1Index, x1Value);
				featureList.add(feature);
				x1i++;
			}
			
			if(x1Index==x2Index){
				Feature feature = new FeatureNode(x1Index, x1Value+x2Value);
				featureList.add(feature);
				x1i++;
				x2i++;
			}
		}
		while(x1i<x1.length){
			Feature x1Feature = x1[x1i];
			int x1Index = x1Feature.getIndex();
			double x1Value = x1Feature.getValue();
			Feature feature = new FeatureNode(x1Index, x1Value);
			featureList.add(feature);
			x1i++;
		}
		while(x2i<x2.length){
			Feature x2Feature = x2[x2i];
			int x2Index = x2Feature.getIndex();
			double x2Value = x2Feature.getValue();
			Feature feature = new FeatureNode(x2Index, x2Value);
			featureList.add(feature);
			x2i++;
		}
		
		Feature[] sumVector = new Feature[featureList.size()];
		for(int i=0;i<featureList.size();i++){
			sumVector[i] = featureList.get(i);
		}
		return sumVector;
	}
	
	
	public static double[] sum(double[] x1, double[] x2){
		int len = x1.length;
		double[] result = new double[len];
		for(int i=0;i<len;i++){
			result[i] = x1[i] + x2[i];
		}
		return result;
	}
	
	public static double[] sub(double[] x1, double[] x2){
		double[] result = new double[x1.length];
		for(int i=0;i<x1.length;i++){
			result[i] = x1[i]- x2[i];
		}
		return result;
	}
	
	public static Feature[] sub(Feature[] x1, Feature[] x2){
		List<Feature> featureList = new ArrayList<Feature>();
		int i1 = 0;
		int i2 = 0;
		while(i1<x1.length && i2<x2.length){
			int index1 = x1[i1].getIndex();
			int index2 = x2[i2].getIndex();
			double value1 = x1[i1].getValue();
			double value2 = x2[i2].getValue();
			if(index1>index2){
				Feature feature = new FeatureNode(index2, -value2);
				featureList.add(feature);
				i2++;
			}
			if(index1<index2){
				Feature feature = new FeatureNode(index1, value1);
				featureList.add(feature);
				i1++;
			}
			if(index1 == index2){
				Feature feature = new FeatureNode(index1, value1-value2);
				featureList.add(feature);
				i1++;
				i2++;
			}
		}
		while(i1<x1.length){
			int index1 = x1[i1].getIndex();
			double value1 = x1[i1].getValue();
			Feature feature = new FeatureNode(index1, value1);
			featureList.add(feature);
			i1++;
		}
		while(i2<x2.length){
			int index2 = x2[i2].getIndex();
			double value2 = x2[i2].getValue();
			Feature feature = new FeatureNode(index2, -value2);
			featureList.add(feature);
			i2++;
		}
		Feature[] result = new Feature[featureList.size()];
		for(int i=0;i<featureList.size();i++){
			result[i] = featureList.get(i);
		}
		return result;
	}
	
	public static double[] expand(Feature[] xi, int n){
		double[] result = new double[n];
		for(Feature feature : xi){
			int index = feature.getIndex();
			double value = feature.getValue();
			result[index] = value;
		}
		return result;
	}
	public static Feature[] compact(double[] w){
		List<Feature> featureList = new ArrayList<Feature>();
		for(int i=0;i<w.length;i++){
			if(w[i]!=0){
				Feature f = new FeatureNode(i, w[i]);
				featureList.add(f);
			}
		}
		Feature[] wc = new Feature[featureList.size()];
		for(int i=0;i<wc.length;i++){
			wc[i] = featureList.get(i);
		}
		return wc;
	}
	
	
}
