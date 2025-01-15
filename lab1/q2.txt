#include <stdio.h>

int main() {
    int rows;
    
    printf("Enter the number of rows: ");
    scanf("%d", &rows);

    printf("Output: \n");

    int spaces = rows - 1;
    int numCharsPerRow = (rows * 2) - 1;

    for (int i = 0; i < rows; i++) {
        int numStarsForThisRow = numCharsPerRow - (spaces * 2);

        for (int s = 0; s < spaces; s++) {
            printf(" ");
        }

        for (int star = 0; star < numStarsForThisRow; star++) {
            printf("*");
        }

        for (int s = 0; s < spaces; s++) {
            printf(" ");
        }

        spaces--;
        printf("\n");
    }

    return 0;
}