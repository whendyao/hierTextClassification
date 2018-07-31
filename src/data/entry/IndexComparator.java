package data.entry;

import java.util.Comparator;

public class IndexComparator implements Comparator<Feature> {

	@Override
	public int compare(Feature o1, Feature o2) {
		if(o1.getIndex()<o2.getIndex()){
			return -1;
		}else if(o1.getIndex()==o2.getIndex()){
			return 0;
		}else{
			return 1;
		}
		
	}


}
