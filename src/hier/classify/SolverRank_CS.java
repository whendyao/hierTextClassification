package hier.classify;

import java.util.Random;

import data.entry.DProblem;
import data.entry.Feature;
import data.entry.vectorOperation;

public class SolverRank_CS {
	public static Random rand = new Random();
	public static double esp = 0.001;
	public static void swap(int[] array, int i,int j){
		int temp = array[i];
		array[i] = array[j];
		array[j] = temp;
	}
	public static void resort(int[] alphArray, int activeSize){
		Random rand = new Random();
		for(int i=0;i<activeSize;i++){
			int j = i+rand.nextInt(activeSize-i);
			SolverRank_CS.swap(alphArray, i, j);
		}
	}
	public static void swap(Object[] array, int i, int j){
		Object temp = array[i];
		array[i] = array[j];
		array[j] = temp;
	}
	
	public static void resort(Object[] array, int size){
		for(int i=0;i<size;i++){
			int index = i + rand.nextInt(size-i);
			SolverRank_CS.swap(array, i, index);
		}
	}
	public static double[] solve(DProblem prob, double c, String labelName){
		double[] yl = prob.callabelVec(labelName);
		int trainNum = prob.trainNum;
		int featureNum = prob.featureNum;
		double[] w = new double[featureNum];
		double[] alpha = new double[trainNum];
		int activeSize = 0;
		int[] activeArray = new int[prob.trainNum];
		for(int i=0;i<prob.trainNum;i++){
			activeArray[i] = i;
		}
		double maxGrad;
		double minGrad;
		double ShrinkUpBound;
		double ShrinklowBound;
		while(activeSize<activeArray.length){
			activeSize = activeArray.length;
			maxGrad = Double.POSITIVE_INFINITY;
			minGrad = Double.NEGATIVE_INFINITY;
			ShrinkUpBound = Double.POSITIVE_INFINITY;
			ShrinklowBound = Double.NEGATIVE_INFINITY;
			while((maxGrad-minGrad)>SolverRank_CS.esp){
				SolverRank_CS.resort(activeArray, activeSize);
				maxGrad = Double.NEGATIVE_INFINITY;
				minGrad = Double.POSITIVE_INFINITY;
				for(int i=0;i<activeSize;i++){
					int index = activeArray[i];
					Feature[] xi = prob.x[index];
					double yil = yl[index];
					double alphail = alpha[index];
					double alphaUpBound = c;
					double alphalowBound = 0;
					double grad = vectorOperation.dotProd(w, xi)*yil-1;
					if(alphalowBound==alphail){
						if(grad>ShrinkUpBound){
							activeSize--;
							SolverRank_CS.swap(activeArray, i, activeSize);
							i--;
							continue;
						}
						if(grad>0){
							grad = 0;
						}
					}
					if(alphail==alphaUpBound){
						if(grad<ShrinklowBound){
							activeSize--;
							SolverRank_CS.swap(activeArray, i, activeSize);
							i--;
							continue;
						}
						if(grad<0){
							grad = 0;
						}
					}
					if(grad>maxGrad){
						maxGrad = grad;
					}
					if(grad<minGrad){
						minGrad = grad;
					}
					if(grad!=0){
						double qii = vectorOperation.dotProd(xi, xi);
						if(qii==0){
							alphail = alphaUpBound;
						}else{
							alphail = alphail - grad/qii;
							if(alphail<alphalowBound){
								alphail = alphalowBound;
							}else if(alphail>alphaUpBound){
								alphail = alphaUpBound;
							}
						}
						double deltaAlpha = alphail - alpha[index];
						SolverRank_CS.updateW(deltaAlpha, xi, yil, w);
						alpha[index] = alphail;
					}
				}
				if(maxGrad>0){
					ShrinkUpBound = maxGrad;
				}else{
					ShrinkUpBound = Double.POSITIVE_INFINITY;
				}
				
				if(minGrad<0){
					ShrinklowBound = minGrad;
				}else{
					ShrinklowBound = Double.NEGATIVE_INFINITY;
				}
			}
		}
		return w;
	}
	
	public static double[] solve(DProblem prob, double c, String labelName, double[] delta){
		double[] yl = prob.callabelVec(labelName);
		int trainNum = prob.trainNum;
		int featureNum = prob.featureNum;
		double[] w = new double[featureNum];
		double[] alpha = new double[trainNum];
		int activeSize = 0;
		int[] activeArray = new int[prob.trainNum];
		for(int i=0;i<prob.trainNum;i++){
			activeArray[i] = i;
		}
		double maxGrad;
		double minGrad;
		double ShrinkUpBound;
		double ShrinklowBound;
		while(activeSize<activeArray.length){
			activeSize = activeArray.length;
			maxGrad = Double.POSITIVE_INFINITY;
			minGrad = Double.NEGATIVE_INFINITY;
			ShrinkUpBound = Double.POSITIVE_INFINITY;
			ShrinklowBound = Double.NEGATIVE_INFINITY;
			while((maxGrad-minGrad)>SolverRank_CS.esp){
				SolverRank_CS.resort(activeArray, activeSize);
				maxGrad = Double.NEGATIVE_INFINITY;
				minGrad = Double.POSITIVE_INFINITY;
				for(int i=0;i<activeSize;i++){
					int index = activeArray[i];
					Feature[] xi = prob.x[index];
					double yil = yl[index];
					double alphail = alpha[index];
					double alphaUpBound = c*delta[index];
					double alphalowBound = 0;
					double grad = vectorOperation.dotProd(w, xi)*yil-1;
					if(alphalowBound==alphail){
						if(grad>ShrinkUpBound){
							activeSize--;
							SolverRank_CS.swap(activeArray, i, activeSize);
							i--;
							continue;
						}
						if(grad>0){
							grad = 0;
						}
					}
					if(alphail==alphaUpBound){
						if(grad<ShrinklowBound){
							activeSize--;
							SolverRank_CS.swap(activeArray, i, activeSize);
							i--;
							continue;
						}
						if(grad<0){
							grad = 0;
						}
					}
					if(grad>maxGrad){
						maxGrad = grad;
					}
					if(grad<minGrad){
						minGrad = grad;
					}
					if(grad!=0){
						double qii = vectorOperation.dotProd(xi, xi);
						if(qii==0){
							alphail = alphaUpBound;
						}else{
							alphail = alphail - grad/qii;
							if(alphail<alphalowBound){
								alphail = alphalowBound;
							}else if(alphail>alphaUpBound){
								alphail = alphaUpBound;
							}
						}
						double deltaAlpha = alphail - alpha[index];
						SolverRank_CS.updateW(deltaAlpha, xi, yil, w);
						alpha[index] = alphail;
					}
				}
				if(maxGrad>0){
					ShrinkUpBound = maxGrad;
				}else{
					ShrinkUpBound = Double.POSITIVE_INFINITY;
				}
				
				if(minGrad<0){
					ShrinklowBound = minGrad;
				}else{
					ShrinklowBound = Double.NEGATIVE_INFINITY;
				}
			}
		}
		return w;
	}
	
	public static double[] solve(DProblem prob, double c, double[] alpha,String labelName){
		double[] yl = prob.callabelVec(labelName);
		int trainNum = prob.trainNum;
		int featureNum = prob.featureNum;
		double[] w = new double[featureNum];
		for(int i=0;i<trainNum;i++){
			if(alpha[i]!=0){
				for(Feature f : prob.x[i]){
					int index = f.getIndex();
					double value = f.getValue();
					w[index] = w[index] + alpha[i]*value*yl[i];
				}
			}
		}
		int activeSize = 0;
		int[] activeArray = new int[prob.trainNum];
		for(int i=0;i<prob.trainNum;i++){
			activeArray[i] = i;
		}
		double maxGrad;
		double minGrad;
		double ShrinkUpBound;
		double ShrinklowBound;
		while(activeSize<activeArray.length){
			activeSize = activeArray.length;
			maxGrad = Double.POSITIVE_INFINITY;
			minGrad = Double.NEGATIVE_INFINITY;
			ShrinkUpBound = Double.POSITIVE_INFINITY;
			ShrinklowBound = Double.NEGATIVE_INFINITY;
			while((maxGrad-minGrad)>SolverRank_CS.esp){
				SolverRank_CS.resort(activeArray, activeSize);
				maxGrad = Double.NEGATIVE_INFINITY;
				minGrad = Double.POSITIVE_INFINITY;
				for(int i=0;i<activeSize;i++){
					int index = activeArray[i];
					Feature[] xi = prob.x[index];
					double yil = yl[index];
					double alphail = alpha[index];
					double alphaUpBound = c;
					double alphalowBound = 0;
					double grad = vectorOperation.dotProd(w, xi)*yil-1;
					if(alphalowBound==alphail){
						if(grad>ShrinkUpBound){
							activeSize--;
							SolverRank_CS.swap(activeArray, i, activeSize);
							i--;
							continue;
						}
						if(grad>0){
							grad = 0;
						}
					}
					if(alphail==alphaUpBound){
						if(grad<ShrinklowBound){
							activeSize--;
							SolverRank_CS.swap(activeArray, i, activeSize);
							i--;
							continue;
						}
						if(grad<0){
							grad = 0;
						}
					}
					if(grad>maxGrad){
						maxGrad = grad;
					}
					if(grad<minGrad){
						minGrad = grad;
					}
					if(grad!=0){
						double qii = vectorOperation.dotProd(xi, xi);
						if(qii==0){
							alphail = alphaUpBound;
						}else{
							alphail = alphail - grad/qii;
							if(alphail<alphalowBound){
								alphail = alphalowBound;
							}else if(alphail>alphaUpBound){
								alphail = alphaUpBound;
							}
						}
						double deltaAlpha = alphail - alpha[index];
						SolverRank_CS.updateW(deltaAlpha, xi, yil, w);
						alpha[index] = alphail;
					}
				}
				if(maxGrad>0){
					ShrinkUpBound = maxGrad;
				}else{
					ShrinkUpBound = Double.POSITIVE_INFINITY;
				}
				
				if(minGrad<0){
					ShrinklowBound = minGrad;
				}else{
					ShrinklowBound = Double.NEGATIVE_INFINITY;
				}
			}
		}
		return w;
	}
	
	
	
	
	public static double[] biasedSolve(DProblem prob, double c, String labelName){
		int trainNum = prob.trainNum;
		int featureNum = prob.featureNum;
		double[] w = new double[featureNum];
		double[] alpha = new double[trainNum];
		double[] yl = prob.callabelVec(labelName);
		int activeSize = 0;
		int[] activeArray = new int[prob.trainNum];
		for(int i=0;i<prob.trainNum;i++){
			activeArray[i] = i;
		}
		double maxGrad;
		double minGrad;
		double ShrinkUpBound;
		double ShrinklowBound;
		while(activeSize<activeArray.length){
			activeSize = activeArray.length;
			maxGrad = Double.POSITIVE_INFINITY;
			minGrad = Double.NEGATIVE_INFINITY;
			ShrinkUpBound = Double.POSITIVE_INFINITY;
			ShrinklowBound = Double.NEGATIVE_INFINITY;
			while((maxGrad-minGrad)>SolverRank_CS.esp){
				SolverRank_CS.resort(activeArray, activeSize);
				maxGrad = Double.NEGATIVE_INFINITY;
				minGrad = Double.POSITIVE_INFINITY;
				for(int i=0;i<activeSize;i++){
					int index = activeArray[i];
					Feature[] xi = prob.x[index];
					double yil = yl[index];
					double alphail = alpha[index];
					double alphaUpBound = c;
					if(yil>0){
						alphaUpBound = Double.POSITIVE_INFINITY;
					}
					double alphalowBound = 0;
					double grad = vectorOperation.dotProd(w, xi)*yil-1;
					if(alphalowBound==alphail){
						if(grad>ShrinkUpBound){
							activeSize--;
							SolverRank_CS.swap(activeArray, i, activeSize);
							i--;
							continue;
						}
						if(grad>0){
							grad = 0;
						}
					}
					if(alphail==alphaUpBound){
						if(grad<ShrinklowBound){
							activeSize--;
							SolverRank_CS.swap(activeArray, i, activeSize);
							i--;
							continue;
						}
						if(grad<0){
							grad = 0;
						}
					}
					if(grad>maxGrad){
						maxGrad = grad;
					}
					if(grad<minGrad){
						minGrad = grad;
					}
					if(grad!=0){
						double qii = vectorOperation.dotProd(xi, xi);
						if(qii==0){
							alphail = alphaUpBound;
						}else{
							alphail = alphail - grad/qii;
							if(alphail<alphalowBound){
								alphail = alphalowBound;
							}else if(alphail>alphaUpBound){
								alphail = alphaUpBound;
							}
						}
						double deltaAlpha = alphail - alpha[index];
						SolverRank_CS.updateW(deltaAlpha, xi, yil, w);
						alpha[index] = alphail;
					}
				}
				if(maxGrad>0){
					ShrinkUpBound = maxGrad;
				}else{
					ShrinkUpBound = Double.POSITIVE_INFINITY;
				}
				
				if(minGrad<0){
					ShrinklowBound = minGrad;
				}else{
					ShrinklowBound = Double.NEGATIVE_INFINITY;
				}
			}
		}
//		SolverRank_CS.checkW(prob, alpha, w, l);
//		double dualVal = SolverRank_CS.calDual(w, alpha);
//		double targetVal = SolverRank_CS.calTarget(prob, w, l, c);
//		System.out.println("target = " + targetVal + " dual = " + dualVal);
		return w;
	}
	
	public static void updateW(double deltaAlpha, Feature[] xi, double yil, double[] w){
		for(Feature feature : xi){
			int index = feature.getIndex(); 
			double value = feature.getValue();
			w[index] = w[index] + deltaAlpha*value*yil;
		}
	}	
	
}
