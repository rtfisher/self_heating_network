      subroutine helm_wrapper (invert, dens, temp, abar, zbar, pres, eint, cs, enthalpy, cp, cv, gammac, gammae)

! This is a simplified wrapper for the Helmholtz EOS. For
! invert = .FALSE., it takes only density, temperature,
! and composition (through abar and zbar) for a single
! cell, and computes the resulting pressures, internal
! energies, entropies, and specific heats, and returns
! these through the common blocks in the include decks
! as well as in the subroutine call itself.

! For invert = .TRUE., pres, temp, abar and zbar are taken
! as arguments, along with an initial guess for dens, and
! density and the other quantities are returned.

      include 'implno.dek'
      include 'const.dek'
      include 'vector_eos.dek'

     
      logical invert ! input 
      double precision temp, abar, zbar ! inputs
      double precision dens, pres ! input or output depending on mode  
      double precision eint, cs, enthalpy, gammac, gammae, cp, cv ! outputs

      temp_row (1) = temp
      den_row (1) = dens
      abar_row (1) = abar
      zbar_row (1) = zbar

! here is the tabular helmholtz free energy eos:
!
! routine read_helm_table reads an electron helm free energy table
! routine read_helm_iontable reads an ion free energy table
! routine helmeos computes the pressure, energy and entropy via tables
! routine helmeos3 uses fermionic ions
! routine helmeos2 adds third derivatives
! routine helmeos_orig as helmeos with the table read inside the routine

      jlo_eos = 1 ; jhi_eos = 1 ! 1 entry only

! Call helmeos or its inversion (invert_helm_pt) depending on whether
!  invert is .TRUE. or .FALSE.

      if (invert) then
        ptot_row (1) = pres
        call invert_helm_pt ! call helm with pres, temp mode
      else 
        call helmeos ! call helm with dens, temp mode
      endif

! Extract output from common blocks

      cs = cs_row (1)
      eint = etot_row (1)

      if ( .not. (invert) ) then
        pres = ptot_row (1)  
      else 
        dens = den_row (1)
      endif

      enthalpy = etot_row (1) + ptot_row (1) / den_row (1)
      cp   = cp_row (1)
      cv   = cv_row (1) 
      gammac = gam1_gas_row(1)
      gammae = gam2_gas_row(1)

      end
