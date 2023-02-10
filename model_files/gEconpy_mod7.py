# --------------------------------------------------------------
# Base Model + Calvo Wages + Taylor rule
# Reference: Costas
# Author: Lukas Grahl
# --------------------------------------------------------------

options
{
    output logfile = FALSE;
    output LaTeX = FALSE;
};

tryreduce
{
   U[], TC[] ;
};

block HOUSEHOLD
{
    definitions
    {
        u[] = C[] ^ (1 - sigma_C) / (1 - sigma_C) - Ls[] ^ (1 + sigma_L) / (1 + sigma_L) ;
    };

    controls
    {
        C[], Ls[], I[], Ks[];
    };

    objective
    {
        U[] = u[] + beta * E[][U[1]] ;
    };

    constraints
    {
        P[] * (C[] + I[]) = R_k[] * Ks[-1] + w[] * Ls[] + Pi[] : lambda[] ;
        Ks[] = (1 - delta) * Ks[-1] + I[] : q[] ;
    };

    identities
    {
        Q[] = q[] / lambda[] ;                                  # Tobin's q
    };

    calibration
    {
        beta = 0.985;
        delta = 0.02;
        sigma_C = 2;
        sigma_L = 1.5;
    };
};

block FIRM
{
    controls
    {
        Kd[], Ld[];
    };

    objective
    {
        TC[] = -(R_k[] * Kd[] + w[] * Ld[]);
    };

    constraints
    {
        Y[] = A[] * Kd[] ^ alpha * Ld[] ^ (1 - alpha) : mc[];
    };

    calibration
    {
        alpha = 0.35;
    };
};

block TECHNOLOGY_SHOCKS
{
    identities
    {
        log(A[]) = rho_A * log(A[-1]) + epsilon_A[];
    };

    shocks
    {
        epsilon_A[] ~ N(mean=0, sd=sigma_epsilon);
    };

    calibration
    {
        rho_A ~ Beta(mean=0.95, sd=0.04) = 0.95;
        sigma_epsilon ~ Inv_Gamma(mean=0.1, sd=0.01) = 0.01;
    };
};

block PRICE_SETTING
# obtain P_hat, the optimal price at pricing signal
{
    identities
    {
        g[] = mc[] + (beta * Theta) * E[][g[1]] ;

        P_hat[] = (psi / (psi - 1)) * g[] ;
        
        P[] = (Theta  * P[-1] ^ (1 - psi) + (1 - Theta) * P_hat[] ^ (1 - psi)) ^ (1 / (1 - psi)) ;

        pi[] = P[] / P[-1] ;
    };

    calibration
    {
        Theta = .75 ;         # Calvo pricing probability
        psi = 8 ;
    };   

};

block WAGE_SETTING
{
    identities
    {
        w_hat[] = (psi_w / (psi_w - 1)) * f[] ;

        f[] = C[] ^ sigma_C * L_d[] ^ sigma_L * P[] + beta * Theta * E[][f[1]] ;

        w[] = (Theta_w * w[-1] ^ (1 - psi_w) + (1 - Theta_w) * w_hat[] ^ (1 - psi_w)) ^ (1 / (1 - psi_w)) ; 

        pi_w[] = w[] / w[-1] ;
    };

    calibration
    {
        Theta_w = .75 ;
        psi_w = 21 ;
    };
};

block GOVERNMENT
{
    identities
    {
#        B[+1] / R_b[] = B[]
    };  
};

block MONETARY_AUTHORTIY
{
    identities
    {
#        R_b[] / R_b[ss] = (R_b[] / R_b[-1])^ gamma_r * ((pi[] / pi[ss]) ^ gamma_pi * (Y[] / Y[ss]) ^ (gamma_y)) ^ (1 - gamma_r) + S_CB[];
    };  

    calibration
    {
#        gamma_pi = .1 ; # preference parameter on deviation from pi_t
#        gamma_y = .1 ; # preference parameter on deviation from y_n
#        gamma_r = .1 ; # preference parameter for change in interest rate
    };
};

block MONETARY_SHOCK
{
    identities
    {
        log(S_CB[]) = rho_CB * log(S_CB[-1]) + epsilon_CB[];
    };  

    shocks
    {
        epsilon_CB[] ~ N(mean=0, sd=sigma_epsilon_CB);
    };

    calibration
    {
        sigma_epsilon_CB ~ Inv_Gamma(mean=0.1, sd=0.01) = 0.01;
    };
};

block EQUILIBRIUM
{
    identities
    {
        Y[] = I[] + C[] ; 
        Kd[] = Ks[] ;
        Ld[] = Ls[] ;
        Pi[] = Y[] - TC[] ;
        R_b[ss] = R_k[ss] ;
    };  
};