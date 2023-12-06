!---------------------------------------------------------------------
!..this file contains routines to invert the helmholtz eos
!..
!..routine invert_helm_pt is used when the pressure and temperature are given
!..routine invert_helm_pt_quiet is as above, but supresses all error messages
!..routine invert_helm_pd is used when the pressure and density are given
!..routine invert_helm_et is used when the energy and temperature are given
!..routine invert_helm_ed is used when the energy and density are given
!..routine invert_helm_st is used when the entropy and temperature are given
!..routine invert_helm_st_quiet is as above, but supresses all error messages



      subroutine invert_helm_pt
      include 'implno.dek'
      include 'const.dek'
      include 'vector_eos.dek'


!..given the pressure, temperature, and composition
!..find everything else

!..it is assumed that ptot_row(j), temp_row(j), abar_row(j), 
!..zbar_row(j), and the pipe limits (jlo_eos:jhi_eos), have 
!..been set before calling this routine.

!..on input den_row(j) conatins a guess for the density,
!..on output den_row(j) contains the converged density.

!..To get the greatest speed advantage, the eos should be fed a
!..large pipe of data to work on.


!..local variables
      integer          i,j,jlo_save,jhi_save
      double precision den,f,df,dennew,eostol,fpmin
      parameter        (eostol = 1.0d-8, &
                       fpmin  = 1.0d-14)


!..initialize
      jlo_save = jlo_eos
      jhi_save = jhi_eos
      do j=jlo_eos, jhi_eos
       eoswrk01(j) = 0.0d0
       eoswrk02(j) = 0.0d0
       eoswrk03(j) = ptot_row(j)
       eoswrk04(j) = den_row(j)
      end do


!..do the first newton loop with all elements in the pipe
      call helmeos 

      do j = jlo_eos, jhi_eos

       f     = ptot_row(j)/eoswrk03(j) - 1.0d0
       df    = dpd_row(j)/eoswrk03(j)
       eoswrk02(j) = f/df

!..limit excursions to factor of two changes
       den    = den_row(j)
       dennew = min(max(0.5d0*den,den - eoswrk02(j)),2.0d0*den)

!..compute the error
       eoswrk01(j)  = abs((dennew - den)/den)

!..store the new density, keep it within the table limits
       den_row(j)  = min(1.0d14,max(dennew,1.0d-11))
      enddo



!..now loop over each element of the pipe individually
      do j = jlo_save, jhi_save

       do i=2,40

        if (eoswrk01(j) .lt. eostol .or. &
           abs(eoswrk02(j)) .le. fpmin) goto 20 

        jlo_eos = j
        jhi_eos = j

        call helmeos 

        f     = ptot_row(j)/eoswrk03(j) - 1.0d0
        df    = dpd_row(j)/eoswrk03(j)
        eoswrk02(j) = f/df

!..limit excursions to factor of two changes
        den    = den_row(j)
        dennew = min(max(0.5d0*den,den - eoswrk02(j)),2.0d0*den)

!..compute the error
        eoswrk01(j)  = abs((dennew - den)/den)

!..store the new density, keep it within the table limits
        den_row(j)  = min(1.0d14,max(dennew,1.0d-11))   

!..end of netwon loop
       end do


!..we did not converge if we land here
      write(6,*) 
      write(6,*) 'newton-raphson failed in routine invert_helm_pt'
      write(6,*) 'pipeline element',j
      write(6,01) 'pwant  =',eoswrk03(j),' temp =',temp_row(j)
 01   format(1x,5(a,1pe16.8))
      write(6,01) 'error =',eoswrk01(j), &
                 '  eostol=',eostol,'  fpmin =',fpmin
      write(6,01) 'den   =',den_row(j),'  denold=',eoswrk04(j)
      write(6,01) 'f/df  =',eoswrk02(j),' f   =',f,    ' df    =',df
      write(6,*) 
      stop 'could not find a density in routine invert_helm_pt'



!..land here if newton loop converged, back for another pipe element
 20    continue
      end do



!..call eos one more time with the converged value of the density

      jlo_eos = jlo_save
      jhi_eos = jhi_save

      call helmeos

      return
      end 

