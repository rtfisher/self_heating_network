      program main

      include 'implno.dek'
      include 'vector_eos.dek'

! tests the eos routine
! 
! ionmax  = number of isotopes in the network
! xmass   = mass fraction of isotope i
! aion    = number of nucleons in isotope i
! zion    = number of protons in isotope i

      logical          invert
      integer          ionmax, narg
      parameter (ionmax = 1)
!      parameter        (ionmax=2)
      double precision xmass(ionmax),aion(ionmax),zion(ionmax),temp,dens,abar,zbar
      double precision pres, eint, cs, enthalpy, cp, cv, gammac, gammae, arg2
      character(len=32) :: arg

! set the mass fractions, z's and a's of the composition

! hydrogen, heliu, and carbon in solar composition (use ionmax = 3)
!      xmass(1) = 0.75d0 ; aion(1)  = 1.0d0  ; zion(1)  = 1.0d0
!      xmass(2) = 0.23d0 ; aion(2)  = 4.0d0  ; zion(2)  = 2.0d0
!      xmass(3) = 0.02d0 ; aion(3)  = 12.0d0 ; zion(3)  = 6.0d0

! 50/50 C/O composition (use ionmax = 2)
!      xmass(1) = 0.5d0 ; aion(1)  = 12.0d0 ; zion(1)  = 6.0d0
!      xmass(2) = 0.5d0 ; aion(2)  = 16.0d0 ; zion(2)  = 8.0d0

! Pure He composition (use ionmax = 1)
!      xmass(1) = 1.0;    aion(1)  = 4.00; zion(1) = 2.0d0

! general procedure to average atomic weight and charge
!      abar   = 1.0d0/sum(xmass(1:ionmax)/aion(1:ionmax))
!      zbar   = abar * sum(xmass(1:ionmax) * zion(1:ionmax)/aion(1:ionmax))

      narg = COMMAND_ARGUMENT_COUNT()

  ! If no arguments, print usage and exit
      if (narg == 0) then
        print *, 'Usage: helm_program <invert> <dens> <temp> <abar> <zbar> <pres>'
        print *, 'Where:'
        print *, '  invert= True/False for inversion or normal mode'
        print *, '  dens  = density rho (g/cm^3) for invert=False'
        print *, '  temp  = temperature T (K)'
        print *, '  abar  = average atomic weight'
        print *, '  zbar  = average atomic number'
        print *, '  pres  = pressure (dyne/cm^2) for invert=True'
        print *, ' Inversion mode True inputs dens (starting guess), temp, abar, zbar, pres and outputs density and other EOS data'
        print *, ' Inversion mode False inputs density, temp, abar, zbar, and outputs pressure and other EOS data'
        stop
      end if

  ! Read arguments and convert to double precision or logical
      if (narg >= 5) then
        call GET_COMMAND_ARGUMENT(1, arg)
        read(arg, *) invert 
        call GET_COMMAND_ARGUMENT(2, arg)
        read(arg, *) dens
        call GET_COMMAND_ARGUMENT(3, arg)
        read(arg, *) temp
        call GET_COMMAND_ARGUMENT(4, arg)
        read(arg, *) abar
        call GET_COMMAND_ARGUMENT(5, arg)
        read(arg, *) zbar
      else
        print *, 'Error: Insufficient arguments.'
        stop
      end if

   ! If in inversion mode, also read in pressure
      if ((invert) .and. (narg >= 6)) then
        call GET_COMMAND_ARGUMENT(6, arg)
        read(arg, *) pres
      end if


! read the data table
      call read_helm_table

! Invoke a simplified subroutine to call EOS
      call helm_wrapper (invert, dens, temp, abar, zbar, pres, eint, cs, enthalpy, cp, cv, gammac, gammae)

! verbose output
#if 0
      print *, "Inputs"
      print *, "======"
      print *, 'Invert = ', invert
      print *, "Composition (abar / zbar) = ", abar, " ", zbar
      if (.not. (invert) ) then
        print *, "density rho (10^7 g/cm^3) = ", dens / 1.e7
      else 
        print *, "pressure p = ", pres
      endif
      print *, "temperature T (10^8 K) = ", temp / 1.e8
      print *, "Outputs"
      print *, "======="
      if (.not. (invert) ) then
        print *, "pressure p = ", pres
      else
        print *, "density rho (10^7 g/cm^3) = ", dens / 1.e7
      endif
      print *, "specific internal energy e = ", eint
      print *, "gammac = ", gammac
      print *, 'gammae = ', gammae
      print *, "specific enthalpy h = e + pv = e + p / rho = ", eint + pres / dens
      print *, "speed of sound cs (km/s) = ", cs / 1.e5
!      print *, "eint + pres / dens = ",  eint + pres / dens
      print *, "cp (erg / K / g) = ", cp
      print *, "cv (erg / K / g) = ", cv
#endif

! Use the Unix stdout interface to output results
      print *, dens, " ", pres, " ", eint, " ", gammac, " ", gammae, " ", eint + pres / dens, " ", cs, " ", cp, " ", cv

      end
