void print_resultsD(int natoms, char *label, double e, double f[], double v[]) {
    printf("%s\n", label);
    printf("Energy = %16.10f\n", e);
    printf("Forces:\n");
    int atom;
    for (atom = 0; atom < natoms; ++atom)
        printf("%16.10f %16.10f %16.10f\n", f[3 * atom], f[3 * atom + 1], f[3 * atom + 2]);
    printf("Virial:\n%16.10f %16.10f %16.10f %16.10f %16.10f %16.10f\n\n", v[0], v[1], v[2], v[3], v[4], v[5]);
}

void print_resultsF(int natoms, char *label, float e, float f[], float v[]) {
    printf("%s\n", label);
    printf("Energy = %16.10f\n", e);
    printf("Forces:\n");
    int atom;
    for (atom = 0; atom < natoms; ++atom)
        printf("%16.10f %16.10f %16.10f\n", f[3 * atom], f[3 * atom + 1], f[3 * atom + 2]);
    printf("Virial:\n%16.10f %16.10f %16.10f %16.10f %16.10f %16.10f\n\n", v[0], v[1], v[2], v[3], v[4], v[5]);
}
