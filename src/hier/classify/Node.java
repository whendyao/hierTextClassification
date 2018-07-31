package hier.classify;

import java.util.HashSet;
import java.util.Set;

public class Node {
	public String parentName;
	public String nodeName;
	public String labelName;
	public double weight;
	public double insNum;
	public int index;
	public double[] w;
	public double[] bw;
	public double[] alpha;
	public int depth;
	public Set<Node> childSet;
	public Node(String nodeName){
		this.nodeName = nodeName;
		this.childSet = new HashSet<Node>();
	}
	
	public void addChild(Node childNode){
		this.childSet.add(childNode);
	}
}
