#include<iostream>
#include<vector>
#include<numeric>
#include<algorithm>


float truePositiveRate(std::vector<int> &y_true, std::vector<float> &y_score, float threshold, int len)
{
    int TP = 0, FN = 0;
    for(int i = 0; i < len; i++)
    {
        if(y_score[i] >= threshold)
        {
            if(y_true[i] == 1)
                TP++;
            else
                FN++;
        }
    }
    std::cout << "TP: " << TP << " FN: " << FN << std::endl;
    return (float)TP / (TP + FN);
}


//Calculate the auc score in multi-label classification
float auc_score(std::vector<std::vector<int>> Y_true, std::vector<std::vector<float>> Y_score)
{
    int n_classes = Y_true.size();
    int n_samples = Y_true[0].size();

    std::vector<float> tpr(n_samples);
    std::vector<float> fpr(n_samples);

    for(int i = 0; i < n_classes; i++)
    {
        std::vector<int> y_true = Y_true[i];
        std::vector<float> y_score = Y_score[i];

        int n_positive = std::accumulate(y_true.begin(), y_true.end(), 0);
        int n_negative = n_samples - n_positive;

        std::vector<int> order(n_samples);
        for(int i = 0; i < n_samples; i++)
        {
            order[i] = i;
        }
        //sorting order in ascending order of y_score
        std::sort(order.begin(), order.end(), [&y_score](int i1, int i2){return y_score[i1] < y_score[i2];});

        std::vector<int> y_true_sorted(n_samples);
        for(int i = 0; i < n_samples; i++)
        {
            y_true_sorted[i] = y_true[order[i]];
        }

        int tp = 0, fp = 0;
        float prev_x = 0, prev_y = 0;
        float area = 0;

        for(int i = 0; i < n_samples; i++)
        {
            float x = 0, y = 0;
            if(y_true_sorted[i] == 1)
            {
                tp++;
                y = (float)tp / n_positive;
                x = (float)fp / n_negative;
            }
            else
            {
                fp++;
                y = (float)tp / n_positive;
                x = (float)fp / n_negative;
            }

            area += (x - prev_x) * (y + prev_y) / 2;
            prev_x = x;
            prev_y = y;
        }

        tpr[i] = prev_y;
        fpr[i] = prev_x;
    }

    float auc = trapezoidalArea(tpr, fpr);
    return auc;
}

float trapezoidalArea(std::vector<float> tpr, std::vector<float> fpr)
{
    int n = tpr.size();
    float area = 0;
    for(int i = 1; i < n; i++)
    {
        area += (fpr[i] - fpr[i - 1]) * (tpr[i] + tpr[i - 1]) / 2;
    }
    return area;
}

/*
float auc_score(std::vector<float> &y_true, std::vector<float> &y_score)
{
    int n = y_true.size();
    std::vector<int> order(n);
    for(int i = 0; i < n; i++)
    {
        order[i] = i;
    }
    //sorting order in ascending order of y_score
    std::sort(order.begin(), order.end(), [&y_score](int i1, int i2){return y_score[i1] < y_score[i2];});

    std::vector<int> y_true_sorted(n);
    for(int i = 0; i < n; i++)
    {
        y_true_sorted[i] = y_true[order[i]];
    }

    int n_positive = std::accumulate(y_true.begin(), y_true.end(), 0);
    int n_negative = n - n_positive;

    int tp = 0, fp = 0;
    float prev_x = 0, prev_y = 0;
    float area = 0;

    for(int i = 0; i < n; i++)
    {
        float x = 0, y = 0;
        if(y_true_sorted[i] == 1)
        {
            tp++;
            y = (float)tp / n_positive;
            x = (float)fp / n_negative;
        }
        else
        {
            fp++;
            y = (float)tp / n_positive;
            x = (float)fp / n_negative;
        }

        area += (x - prev_x) * (y + prev_y) / 2;
        prev_x = x;
        prev_y = y;
    }

    return area;
}
*/


float trapezoidalArea(std::vector<float>& P, float delta = 0.05)
{
    int n = static_cast<int>(1 / delta) + 1;
    float area = P.front() + P.back();

    for(int i = 1; i < n - 1; i++)
    {
        area += 2 * P[i];
    }

    area = area * delta / 2;
    return area;
}

int main()
{
    std::vector<float> y_true = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    std::vector<float> y_score = {0.82714215, 0.60445959, 0.79163409, 0.16080518, 0.61122292,
       0.25550873, 0.56815077, 0.0599057 , 0.66444341, 0.11293577,
       0.06152372, 0.35250697, 0.32267018, 0.43339115, 0.22807443,
       0.72198484, 0.23527699, 0.28502453, 0.41070479, 0.20083562,
       0.37119218, 0.42348227, 0.4876482 , 0.42348227, 0.57509852,
       0.67340477, 0.73558926, 0.71378991, 0.38739725, 0.24042033,
       0.16634116, 0.16634116, 0.28502453, 0.36837418, 0.17375785,
       0.43636291, 0.72198484, 0.46745878, 0.23527699, 0.17202866,
       0.17786914, 0.4433536 , 0.27685038, 0.06891755, 0.21414011,
       0.27120595, 0.26328217, 0.48056205, 0.0884856 , 0.25550873,
       0.56815077, 0.28502453, 0.84221642, 0.52808205, 0.63027285,
       0.93251628, 0.06222562, 0.8823445 , 0.67073977, 0.89166341,
       0.64892548, 0.55521198, 0.75102755, 0.23310831, 0.29334213,
       0.60445959, 0.63027285, 0.9585115 , 0.93428007, 0.32267018,
       0.79823018, 0.22102863, 0.9390781 , 0.50787801, 0.73793446,
       0.87500786, 0.47047017, 0.4433536 , 0.56518147, 0.8658845 ,
       0.89702461, 0.9712638 , 0.56518147, 0.51798738, 0.4038554 ,
       0.943547  , 0.57805065, 0.59474492, 0.39704329, 0.79163409,
       0.72198484, 0.79163409, 0.28502453, 0.76585136, 0.73793446,
       0.71378991, 0.4876482 , 0.63027285, 0.5310945 , 0.35250697};

    
    std::cout << auc_score(y_true, y_score) << std::endl;

    return 0;
}