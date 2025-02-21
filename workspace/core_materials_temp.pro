Include "Parameter.pro";
Function{
  b = {0.0, 0.01461935, 0.0292387, 0.043858049999999996, 0.0584774, 0.07309675, 0.08771609999999999, 0.10233545} ;
  mu_real = {2977.803127202447, 3080.4595557349644, 3182.6638859872783, 3283.347956860391, 3404.225773227987, 3568.0439714454315, 3725.965554037814, 3852.882139839328} ;
  mu_imag = {144.42229082225015, 161.21550049514175, 178.51471340616237, 195.14463770349258, 230.09988245110668, 305.71081373575265, 398.53894480925976, 484.87479873033885} ;
  mu_imag_couples = ListAlt[b(), mu_imag()] ;
  mu_real_couples = ListAlt[b(), mu_real()] ;
  f_mu_imag_d[] = InterpolationLinear[Norm[$1]]{List[mu_imag_couples]};
  f_mu_real_d[] = InterpolationLinear[Norm[$1]]{List[mu_real_couples]};
  f_mu_imag[] = f_mu_imag_d[$1];
  f_mu_real[] = f_mu_real_d[$1];
 }  