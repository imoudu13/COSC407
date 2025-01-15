#include <stdio.h>
#include <math.h>

int main() {
    int nums[4];
    printf("Enter 4 integers seperated by spaces: ");
    scanf("%d %d %d %d", &nums[0], &nums[1], &nums[2], &nums[3]);

    float avg = (nums[0] + nums[1] + nums[2] + nums[3]) / 4.0;

    int count = 0;

    for ( int i = 0; i < 4; i++) {
        if (nums[i] > avg) {
            count++;
        }
    }

    double rounded_avg = ceil(avg * 10) / 10;
    
    if (count > 1) {
        printf("There are %d entries above the average (%.1f)", count, rounded_avg);
        return 0;
    }

    printf("There is %d entry above the average (%f)", count, rounded_avg);

    return 0;
}