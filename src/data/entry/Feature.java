package data.entry;

/**
 * @since 1.9
 */
public interface Feature {

    int getIndex();

    double getValue();

    void setValue(double value);
    
    void setIndex(int index);
    
    public Feature Copy();
}
